//
//  PTZRegressorBuilder.cpp
//  Relocalization
//
//  Created by jimmy on 5/4/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "PTZRegressorBuilder.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "dt_util.hpp"
#include "cvx_tensor.h"

using std::cout;
using std::endl;
using Eigen::Vector3d;

//Build model from data
bool PTZRegressorBuilder::buildModel(PTZRegressor& model,
                                     const vector<PTZLearningSample> & samples,
                                     const vector<Eigen::Tensor<unsigned char, 3> > & rgb_images,
                                     const char *model_file_name) const
{
    vector<PTZTree *> trees;
    const unsigned int N = (unsigned int)samples.size();
    
    tree_param_.printSelf();
    const int tree_num = tree_param_.tree_num_;
    for (unsigned int i = 0; i<tree_num; i++) {
        vector<unsigned int> training_data_indices;
        vector<unsigned int> outof_bag_data_indices;
        
        PTZRegressorBuilder::outof_bag_sampling(N, training_data_indices, outof_bag_data_indices);
        
        PTZTree *tree = new PTZTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(samples, training_data_indices, rgb_images, tree_param_);
        printf("build a tree cost %lf seconds\n", (clock()-tt)/CLOCKS_PER_SEC );
        trees.push_back(tree);        
               
        // training error
        vector<Eigen::VectorXd > training_errors;
        for (int j = 0; j<training_data_indices.size(); j++) {
            int index = training_data_indices[j];
            PTZTestingResult pred;
            pred.img2d_ = samples[index].img2d_;
            pred.sampled_color_ = samples[index].color_;
            pred.gt_ptz_ = samples[index].ptz_;
            bool isPredict = tree->predict(rgb_images[samples[index].image_index_], pred);
            if (isPredict) {
                Eigen::Vector3d dif = pred.prediction_error();
                training_errors.push_back(dif);
            }
        }
        // validation error
        vector<Eigen::VectorXd> validation_errors;
        for (int j = 0; j<outof_bag_data_indices.size(); j++) {
            int index = outof_bag_data_indices[j];
            PTZTestingResult pred;
            pred.img2d_ = samples[index].img2d_;
            pred.sampled_color_ = samples[index].color_;
            pred.gt_ptz_ = samples[index].ptz_;
            bool isPredict = tree->predict(rgb_images[samples[index].image_index_], pred);
            if (isPredict) {
                auto dif = pred.prediction_error();
                validation_errors.push_back(dif);
            }
        }
        
        Eigen::VectorXd train_error_mean, train_error_median;
        Eigen::VectorXd validation_error_mean, validation_error_median;
        
        DTUtil::meanMedianError<Eigen::VectorXd>(training_errors, train_error_mean, train_error_median);
        DTUtil::meanMedianError<Eigen::VectorXd>(validation_errors, validation_error_mean, validation_error_median);
       
        printf("tree index is %d \n", i);
        printf("predicted training            percentage: %f median error: %f %f %f\n", 1.0*training_errors.size()/training_data_indices.size(),
               train_error_median[0], train_error_median[1], train_error_median[2]);
        printf("predicted outOfBag validation percentage: %f median error: %f %f %f\n", 1.0*validation_errors.size()/outof_bag_data_indices.size(),
               validation_error_median[0], validation_error_median[1], validation_error_median[2]);
        
        
        if (model_file_name != NULL) {
            model.trees_ = trees;
            model.save(model_file_name);
            printf("saved %s\n", model_file_name);
        }
    }
    model.trees_ = trees;
    return true;
}

// build model from subset of images without dropout

bool PTZRegressorBuilder::buildModel(PTZRegressor & model,
                                     const vector<string> & rgb_img_files,
                                     const vector<Eigen::Vector3d > & ptzs,
                                     const char *model_file_name) const
{
    assert(rgb_img_files.size() == ptzs.size());
    
    vector<PTZTree *> trees;
    tree_param_.printSelf();
    
    const int frame_num = (int)rgb_img_files.size();
    const int sampled_frame_num = std::min((int)rgb_img_files.size(), tree_param_.max_frame_num_);
    const int tree_num = tree_param_.tree_num_;
    const int num_random_sample_per_frame = tree_param_.sampler_num_per_frame_;
    const bool is_use_depth = tree_param_.is_use_depth_;
    assert(is_use_depth == false);
    Eigen::AlignedBox<double, 2> invalid_region = PTZTreeUtil::highschool_RS_invalid_region();
    
    for (int n = 0; n<tree_num; n++) {
        // randomly sample frames
        vector<string> sampled_rgb_files;
        vector<Eigen::Vector3d > sampled_ptzs;
        for (int j = 0; j<sampled_frame_num; j++) {
            int index = rand()%frame_num;
            sampled_rgb_files.push_back(rgb_img_files[index]);
            sampled_ptzs.push_back(ptzs[index]);
        }
        
        printf("training from %lu frames\n", sampled_rgb_files.size());
        // sample from selected frames
        vector<Eigen::Tensor<unsigned char, 3> > rgb_images;
        vector<PTZLearningSample> samples;
        for (int j = 0; j<sampled_rgb_files.size(); j++) {
            cv::Mat cur_img = cv::imread(sampled_rgb_files[j].c_str());
            assert(!cur_img.empty());
            
            Eigen::Tensor<unsigned char, 3> cur_img_eigen;
            cvx::cv2eigen(cur_img, cur_img_eigen);
            
            vector<PTZLearningSample> cur_samples;
            cur_samples = PTZTreeUtil::generateLearningSamples(cur_img_eigen,
                                                               sampled_ptzs[j],
                                                               j,
                                                               num_random_sample_per_frame,
                                                               invalid_region);
            
            samples.insert(samples.end(), cur_samples.begin(), cur_samples.end());
            rgb_images.push_back(cur_img_eigen);
        }
        
        printf("training sample number is %lu\n", samples.size());
        
        vector<unsigned int> training_data_indices;
        for (int j = 0; j<samples.size(); j++) {
            training_data_indices.push_back(j);
        }
        assert(training_data_indices.size() == samples.size());
        
        PTZTree *tree = new PTZTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(samples, training_data_indices, rgb_images, tree_param_);
        printf("build a tree cost %lf seconds\n", (clock()-tt)/CLOCKS_PER_SEC );
        trees.push_back(tree);
        
        model.trees_ = trees;
        if (model_file_name != NULL) {            
            model.save(model_file_name);
            printf("saved %s\n", model_file_name);
        }
        
        this->validation_error(model, rgb_img_files, ptzs, 50);
    }
    
    model.trees_ = trees;
    return true;
}




void PTZRegressorBuilder::outof_bag_sampling(const unsigned int N,
                               vector<unsigned int> & bootstrapped,
                               vector<unsigned int> & outof_bag)
{
    vector<bool> isPicked(N, false);
    for (int i = 0; i<N; i++) {
        int rnd = rand()%N;
        bootstrapped.push_back(rnd);
        isPicked[rnd] = true;
    }
    
    for (int i = 0; i<N; i++) {
        if (!isPicked[i]) {
            outof_bag.push_back(i);
        }
    }
}

bool PTZRegressorBuilder::validation_error(const PTZRegressor & model,
                                           const vector<string> & rgb_img_files,
                                           const vector<Eigen::Vector3d > & ptzs,
                                           const int sampled_frame_num) const
{
    assert(rgb_img_files.size() == ptzs.size());
    const int frame_num = (int)rgb_img_files.size();
    const int num_random_sample_per_frame = tree_param_.sampler_num_per_frame_;
    Eigen::AlignedBox<double, 2> invalid_region = PTZTreeUtil::highschool_RS_invalid_region();
    
    // sample testing files
    vector<string> sampled_rgb_files;
    vector<Eigen::Vector3d> sampled_ptzs;
    for (int i = 0; i<sampled_frame_num; i++) {
        int index = rand()%frame_num;
        sampled_rgb_files.push_back(rgb_img_files[index]);
        sampled_ptzs.push_back(ptzs[index]);
    }
    
    // test on sampled files
    for (int i = 0; i<sampled_rgb_files.size(); i++) {
        cv::Mat cur_img = cv::imread(sampled_rgb_files[i].c_str());
        Eigen::Vector3d cur_ptz = sampled_ptzs[i];
        assert(cur_img.channels()  == 3);
        
        Eigen::Tensor<unsigned char, 3> cur_img_eigen;
        cvx::cv2eigen(cur_img, cur_img_eigen);
        
        Eigen::Vector2d pp(cur_img.cols/2.0, cur_img.rows/2.0);
        // sample testing pixel locations
        vector<PTZLearningSample> cur_samples;
        cur_samples = PTZTreeUtil::generateLearningSamples(cur_img_eigen,
                                                           cur_ptz,
                                                           i,
                                                           num_random_sample_per_frame,
                                                           invalid_region);
        // validatin error for each pixel
        vector<Eigen::VectorXd > validation_errors;
        vector<PTZTestingResult> predictions;
        for (int j = 0; j<cur_samples.size(); j++) {
            PTZTestingResult tr;
            tr.img2d_ = cur_samples[j].img2d_;
            tr.sampled_color_ = cur_samples[j].color_;
            bool is_predict = false;
            if (tree_param_.is_average_) {
                is_predict = model.predictAverage(cur_img_eigen, tr);
            }
            else{
                is_predict = model.predictByColor(cur_img_eigen, tr);
            }
            
            if (is_predict) {
                tr.gt_ptz_ = cur_samples[j].ptz_;
                validation_errors.push_back(tr.prediction_error());
                predictions.push_back(tr);
            }
        }
       
        Eigen::VectorXd validation_error_mean, validation_error_median;
        DTUtil::meanMedianError<Eigen::VectorXd>(validation_errors, validation_error_mean, validation_error_median);
        printf("predict %lu from %lu samplers!\n", validation_errors.size(), cur_samples.size());
        cout<<"median validation pixel PTZ error is "<<validation_error_median<<endl;
        
        
        // validation error for an image
        vector<Eigen::VectorXd > validation_ptz_error;
        for (int j = 0; j<predictions.size(); j++) {
            PTZTestingResult pred = predictions[j];
            Eigen::Vector3d estimated_ptz;
            Eigen::Vector2d ref_pt(pred.img2d_[0], pred.img2d_[1]);
            PTZTreeUtil::panTiltFromReferencePointDecodeFocalLength(ref_pt, pred.predict_ptz_, pp, estimated_ptz);
            validation_ptz_error.push_back(cur_ptz - estimated_ptz);
        }
        
        Eigen::VectorXd validation_ptz_error_mean, validation_ptz_error_median;
        DTUtil::meanMedianError<Eigen::VectorXd>(validation_ptz_error, validation_ptz_error_mean, validation_ptz_error_median);
        cout<<"median validation PTZ error is "<<validation_ptz_error_median<<endl<<endl;
        
    }
    return true;
}
