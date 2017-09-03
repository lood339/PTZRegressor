//
//  main.cpp
//  Relocalization
//
//  Created by jimmy on 4/15/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 0
#include <iostream>
#include <string>
#include <vector>
#include "PTZRegressor.h"
#include "PTZRegressorBuilder.h"
#include "PTZTreeUtil.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "cvx_tensor.h"
#include "dt_util.hpp"
#include "ptz_pose_estimation.h"
#include "mat_io.hpp"

using std::cout;
using std::endl;
using Eigen::Vector2d;
using Eigen::Vector3d;



static void help()
{
    printf("program  RFModelFile sequenceParam reprojThreshold distanceThreshold numSample ransacNum saveFile \n");
    printf("PTZ_test rf.txt      seq_param.txt 2               25                5000      500       result.mat   \n");
    printf("PTZ camera pose estimation using random RGB features\n");
    printf("parameter fits to US high school basketball dataset\n");
    printf("distanceThreshold: RGB color distance \n");
}

static void read_testing_sequence_files(const char *file_name,
                                        string & sequence_name,
                                        string & image_sequence_base_dir)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not open %s\n", file_name);
        return;
    }
    
    {
        char buf1[1024] = {NULL};
        char buf2[1024] = {NULL};
        int ret = fscanf(pf, "%s %s", buf1, buf2);
        assert(ret == 2);
        sequence_name = string(buf2);
    }
    
    {
        char buf1[1024] = {NULL};
        char buf2[1024] = {NULL};
        int ret = fscanf(pf, "%s %s", buf1, buf2);
        assert(ret == 2);
        image_sequence_base_dir = string(buf2);
    }
    printf("sequence name: %s\n", sequence_name.c_str());
    printf("image sequence base dir: %s\n", image_sequence_base_dir.c_str());
    
    fclose(pf);
}

int main(int argc, const char * argv[])
{
    if (argc != 8) {
        printf("argc is %d, should be 8 \n", argc);
        help();
        return -1;
    }
    const char * model_file = argv[1];
    const char * sequence_param = argv[2];
    const double reprojection_error_threshold = strtod(argv[3], NULL);
    const double distance_threshold = strtod(argv[4], NULL);
    const int sample_number = (int)strtod(argv[5], NULL);
    const int ransac_num = (int)strtod(argv[6], NULL);
    const char * save_file = argv[7];
    
    /*
    const char * model_file = "/Users/jimmy/Desktop/bmvc16_ptz_calib/model/seq1.txt";
    const char * sequence_param = "/Users/jimmy/Desktop/bmvc16_ptz_calib/seq1_test.txt";
    const double reprojection_error_threshold = 2;
    const double distance_threshold = 25;
    const int sample_number = 5000;
    const char * save_file = "result.mat";
     */
    
    string sequence_file_name;
    string image_sequence_base_dir;
    read_testing_sequence_files(sequence_param, sequence_file_name, image_sequence_base_dir);
    
    
    // load model
    PTZRegressor model;
    bool is_read = model.load(model_file);
    if (!is_read) {
        printf("Error: can not read from model file %s\n", model_file);
        return -1;
    }
    
    // read testing images and ground truth ptz
    vector<string> image_files;
    vector<Eigen::Vector3d > ptzs;
    PTZTreeUtil::read_sequence_data(sequence_file_name.c_str(), image_sequence_base_dir.c_str(), image_files, ptzs);
    Eigen::AlignedBox<double, 2> invalid_region = PTZTreeUtil::highschool_RS_invalid_region();
    
    
    
    Eigen::Vector2d pp(1280.0/2.0, 720.0/2.0);
    ptz_pose_opt::PTZPreemptiveRANSACParameter param;
    param.reprojection_error_threshold_ = reprojection_error_threshold;
    param.sample_number_ = ransac_num;
    
    Eigen::MatrixXd gt_ptz_all(image_files.size(), 3);
    Eigen::MatrixXd estimated_ptz_all(image_files.size(), 3);
    
    
    for (int i = 0; i<image_files.size(); i++) {
        // sample from the image
        cv::Mat cur_img = cv::imread(image_files[i].c_str());
        assert(!cur_img.empty());
        assert(cur_img.cols == 1280 && cur_img.rows == 720);
        
        Eigen::Tensor<unsigned char, 3> cur_img_eigen;
        cvx::cv2eigen(cur_img, cur_img_eigen);
        
        Eigen::Vector3d gt_ptz = ptzs[i];
        
        double tt = clock();
        vector<PTZLearningSample> cur_samples;
        cur_samples = PTZTreeUtil::generateLearningSamples(cur_img_eigen, gt_ptz, i, sample_number, invalid_region);
        //printf("sampler number is %lu\n", cur_samples.size());
        
        // predict each pixel PTZ
        
        vector<Eigen::Vector2d> image_points;
        vector<vector<Eigen::Vector2d> > candidate_pan_tilt;
        for (int j = 0; j<cur_samples.size(); j++) {
            PTZTestingResult tr;
            tr.img2d_ = cur_samples[j].img2d_;
            tr.sampled_color_ = cur_samples[j].color_;
            
            vector<Eigen::Vector3d> ptz_predictions;
            vector<double> dists;
            vector<Eigen::Vector2d> pt_predictions;
            model.predictAll(cur_img_eigen, tr, ptz_predictions, dists);
            assert(ptz_predictions.size() > 0);
            
            for(int k = 0; k<ptz_predictions.size(); k++) {
                if (dists[k] < distance_threshold) {
                    pt_predictions.push_back(Vector2d(ptz_predictions[k].x(), ptz_predictions[k].y()));
                }
            }
            if (pt_predictions.size() > 0) {
                image_points.push_back(Vector2d(cur_samples[j].img2d_.x(), cur_samples[j].img2d_.y()));
                candidate_pan_tilt.push_back(pt_predictions);
            }
        }
        assert(image_points.size() == candidate_pan_tilt.size());
        printf("valid sample number is %lu\n", image_points.size());
        printf("Prediction and camera pose estimation cost time: %f seconds.\n", (clock() - tt)/CLOCKS_PER_SEC);
        
        Eigen::Vector3d estimated_ptz(0, 0, 0);
        bool is_opt = ptz_pose_opt::preemptiveRANSACOneToMany(image_points,
                                                              candidate_pan_tilt,
                                                              pp,
                                                              param, estimated_ptz, false);
        
        if (is_opt) {
            cout<<"ptz estimation error: "<<(gt_ptz - estimated_ptz).transpose()<<endl;
        }
        else {
            printf("Optimize PTZ failed\n");
        }
        gt_ptz_all.row(i) = gt_ptz;
        estimated_ptz_all.row(i) = estimated_ptz;
    }
    std::vector<std::string> var_name;
    std::vector<Eigen::MatrixXd> data;
    var_name.push_back("ground_truth_ptz");
    var_name.push_back("estimated_ptz");
    data.push_back(gt_ptz_all);
    data.push_back(estimated_ptz_all);
    matio::writeMultipleMatrix(save_file, var_name, data);
    
    // simple statistics
    // void meanMedianError(const vector<T> & errors, T & mean, T & median);
    vector<Eigen::VectorXd> errors;
    for (int r = 0; r<gt_ptz_all.rows(); r++) {
        Eigen::VectorXd err = estimated_ptz_all.row(r) - gt_ptz_all.row(r);
        errors.push_back(err);
    }
    
    Eigen::VectorXd mean_error, median_error;
    DTUtil::meanMedianError(errors, mean_error, median_error);
    cout<<"PTZ mean   error: "<<mean_error.transpose()<<endl;
    cout<<"PTZ median error: "<<median_error.transpose()<<endl;
     
 
    return 0;
}

#endif


