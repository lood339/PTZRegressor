//
//  PTZRegressor.cpp
//  Relocalization
//
//  Created by jimmy on 5/4/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "PTZRegressor.h"
#include <Eigen/Dense>
#include "cvx_util.hpp"

using Eigen::Vector3d;

PTZRegressor::PTZRegressor()
{
    
}
PTZRegressor::~PTZRegressor()
{
    
}

bool PTZRegressor::predictAverage(const Eigen::Tensor<unsigned char, 3> & rgb_image,
                                  PTZTestingResult & predict) const
{
    assert(trees_.size() > 0);
    
    PTZTestingResult init_pred = predict;
    Eigen::Vector3d predicted_ptz(0, 0, 0);
    Eigen::Vector3d predicted_color(0, 0, 0);
    int num_valid_pred = 0;
    for (int i = 0; i<trees_.size(); i++) {
        PTZTestingResult tr = init_pred;
        bool isPredict = trees_[i]->predict(rgb_image, tr);
        if (isPredict) {
            predicted_ptz   += tr.predict_ptz_;
            predicted_color += tr.predict_color_;
            num_valid_pred++;
        }
    }
    if (num_valid_pred > 0) {
        predict.predict_ptz_   = predicted_ptz/(double)num_valid_pred;
        predict.predict_color_ = predicted_color/(double)num_valid_pred;
    }
    
    return num_valid_pred > 0;
}


bool PTZRegressor::predictByColor(const Eigen::Tensor<unsigned char, 3> & rgb_image,
                                  PTZTestingResult & predict) const
{
    assert(trees_.size() > 0);
    
    double min_color_dis = INT_MAX;
    bool is_valid = false;
    PTZTestingResult init_pred = predict;
    for (int i = 0; i<trees_.size(); i++) {
        PTZTestingResult tr = init_pred;
        bool isPredict = trees_[i]->predict(rgb_image, tr);
        if (isPredict) {
            Eigen::Vector3d dif = tr.sampled_color_ - tr.predict_color_;
            double dis = dif.norm();
            // select the one with minimum color distance
            if (dis < min_color_dis) {
                min_color_dis = dis;
                predict = tr;
                is_valid = true;
            }
        }
    }
    return is_valid;
}

bool PTZRegressor::predictAll(const Eigen::Tensor<unsigned char, 3> & rgb_image,
                              PTZTestingResult & sample,
                              vector<Eigen::Vector3d> & predictions,
                              vector<double> & dists)
{
    assert(trees_.size() > 0);
    
    PTZTestingResult init_pred = sample;
    vector<Eigen::Vector3d> unordered_predictions;
    vector<double> unordered_dists;
    for (int i = 0; i<trees_.size(); i++) {
        PTZTestingResult tr = init_pred;
        bool isPredict = trees_[i]->predict(rgb_image, tr);
        if (isPredict) {
            unordered_predictions.push_back(tr.predict_ptz_);
            unordered_dists.push_back((init_pred.sampled_color_ - tr.predict_color_).norm());
        }
    }
    assert(unordered_predictions.size() == unordered_dists.size());
    
    // sort by distance
    vector<size_t> sortIndexes = CvxUtil::sortIndices<double>(unordered_dists);
    for (int i = 0; i<sortIndexes.size(); i++) {
        predictions.push_back(unordered_predictions[sortIndexes[i]]);
        dists.push_back(unordered_dists[sortIndexes[i]]);
    }
    
    assert(predictions.size() == dists.size());
    
    return predictions.size() > 0;
}

bool PTZRegressor::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    //write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    PTZTreeParameter param = trees_[0]->param_;
    param.writeToFile(pf);
    
    vector<string> tree_files;
    string baseName = string(fileName);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".txt");
        fprintf(pf, "%s\n", fileName.c_str());
        tree_files.push_back(fileName);
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        PTZTreeNode::writeTree(tree_files[i].c_str(), trees_[i]->rootNode());
    }
    
    fclose(pf);
    printf("save to PTZ Regressor model %s\n", fileName);    
    return true;
}

bool PTZRegressor::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    PTZTreeParameter param;
    bool is_read = param.readFromFile(pf);
    assert(is_read);
    
    vector<string> treeFiles;
    for (int i = 0; i<param.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        delete trees_[i];
        trees_[i] = 0;
    }
    trees_.clear();
    
    
    for (int i = 0; i<treeFiles.size(); i++) {
        PTZTreeNode * root = NULL;
        bool isRead = PTZTreeNode::readTree(treeFiles[i].c_str(), root);
        assert(isRead);
        PTZTree *tree = new PTZTree();
        tree->setRootNode(root);
        tree->setTreeParameter(param);
        trees_.push_back(tree);
    }
    printf("read from %s\n", fileName);
    return true;
}