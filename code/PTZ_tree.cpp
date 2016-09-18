//
//  PTZ_tree.cpp
//  Relocalization
//
//  Created by jimmy on 4/16/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "PTZ_tree.h"
#include <algorithm>
#include <limits>
#include <vnl/vnl_random.h>
#include <iostream>

using std::cout;
using std::endl;

bool PTZTree::buildTree(const vector<PTZLearningSample> & samples,
                        const vector<unsigned int> & indices,
                        const vector<vil_image_view<unsigned char> > & rgbImages,
                        const PTZTreeParameter & param)
{
    assert(indices.size() <= samples.size());
    root_ = new PTZTreeNode(0);
    param_ = param;
    this->configureNode(samples, rgbImages, indices, 0, root_);
    return true;
}


static vector<double> getRandomCadidate(const double min_val, const double max_val, int num)
{
    assert(min_val <= max_val);
    assert(num > 0);
    
    vector<double> data;
    vnl_random rnd;
    for (int i = 0; i<num; i++) {
        data.push_back(rnd.drand32(min_val, max_val));
    }
    return data;
}

static double bestSplitCandidate(const vector<PTZLearningSample> & samples,
                                 const vector<vil_image_view<unsigned char> > & rgb_images,
                                 const vector<unsigned int> & indices,
                                 const PTZSplitParameter & split_param,
                                 const int split_num,
                                 const int min_leaf_node,
                                 vector<unsigned int> & left_indices,
                                 vector<unsigned int> & right_indices,
                                 double & split_value)
{
    
    vector<double> features;
    const int c1 = split_param.channle_[0];
    const int c2 = split_param.channle_[1];
    const vnl_vector_fixed<int, 2> offset = split_param.offset2_;
    const vnl_vector_fixed<double, 2> wt  = split_param.wt_;
    for (int i = 0; i<indices.size(); i++) {
        const int index = indices[i];
        const int img_idx = samples[index].image_index_;
        const int width  = rgb_images[img_idx].ni();
        const int height = rgb_images[img_idx].nj();
        
        vnl_vector_fixed<int, 2> p1 = samples[index].img2d_;
        vnl_vector_fixed<int, 2> p2 = p1 + offset;
        
        int val1 = rgb_images[img_idx](p1[0], p1[1], c1);
        int val2 = 0;
        
        
        if (PTZTreeUtil::is_inside_image(p2[0], p2[1], width, height)) {
            val2 = rgb_images[img_idx](p2[0], p2[1], c2);
        }
        else
        {
            val2 = 0; // background random noise
        }
        
        double feat = wt[0] * val1 + wt[1] * val2;
        features.push_back(feat);
    }
    assert(features.size() == indices.size());
    
    double val_min = *std::min_element(features.begin(), features.end());
    double val_max = *std::max_element(features.begin(), features.end());
    
    vector<double> split_candidate = getRandomCadidate(val_min, val_max, split_num);
    
    const vnl_vector_fixed<double, 3> ptz_wt(1.0, 1.0, 0.001);
    double loss = std::numeric_limits<double>::max();
    for (int i = 0; i<split_candidate.size(); i++) {
        double cur_split = split_candidate[i];
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        
        for (int j = 0; j<features.size(); j++) {
            int index = indices[j];
            if (features[j] < cur_split) {
                cur_left_indices.push_back(index);
            }
            else{
                cur_right_indices.push_back(index);
            }
        }
        if (cur_left_indices.size() * 2 < min_leaf_node || cur_right_indices.size() * 2 < min_leaf_node) {
            continue;
        }
        
        double left_loss = PTZTreeUtil::variance(samples, cur_left_indices, ptz_wt);
        if (left_loss > loss) {
            continue;
        }
        double right_loss = PTZTreeUtil::variance(samples, cur_right_indices, ptz_wt);
        double cur_loss = left_loss + right_loss;
        if (cur_loss < loss) {
            loss = cur_loss;
            left_indices  = cur_left_indices;
            right_indices = cur_right_indices;
            split_value = cur_split;
        }
    }
    
    return loss;
}
                          

bool PTZTree::configureNode(const vector<PTZLearningSample> & samples,
                            const vector<vil_image_view<unsigned char> > & rgb_images,
                            const vector<unsigned int> & indices,
                            int depth,
                            PTZTreeNode *node)
{
    const int max_depth = param_.max_depth_;
    const int min_leaf_node = param_.min_leaf_node_;
    // depth and min leaf node constraint
    if (depth > max_depth || indices.size() <= param_.min_leaf_node_) {
        node->depth_ = depth;
        node->is_leaf_ = true;
        node->sample_num_ = (int)indices.size();
        PTZTreeUtil::mean_std(samples, indices, node->ptz_, node->stddev_);
        vector<vnl_vector_fixed<double, 3>> sample_colors;
        for (int i = 0; i<indices.size(); i++) {
            sample_colors.push_back(samples[indices[i]].color_);
        }
        
        PTZTreeUtil::mean_std(sample_colors, node->color_mu_, node->color_sigma_);
        if (param_.verbose_) {
            //cout<<"pan tilt zoom is "<<node->ptz_<<endl;
            printf("depth, num_leaf_node, %d, %lu\n", depth, indices.size());
            cout<<"pan tilt zoom "<<node->ptz_<<endl;
            cout<<"pan tilt zoom standard deviation is "<<node->stddev_<<endl;
            cout<<"mean color "<<node->color_mu_<<endl;
            cout<<"color stddev "<<node->color_sigma_<<endl;
        }
        return true;
    }
    
    int max_pixel_offset           = param_.max_pixel_offset_;  // in pixel
    int pixel_offset_candidate_num = param_.pixel_offset_candidate_num_;  // large number less randomness
    int split_candidate_num  = param_.split_candidate_num_;  // number of split in [v_min, v_max]
    int weight_candidate_num = param_.weight_candidate_num_;
    
    vector<vnl_vector_fixed<double, 2> > random_weights;
    random_weights.push_back(vnl_vector_fixed<double, 2>(1.0, -1.0));
    vnl_random rnd;
    for (int i = 0; i<weight_candidate_num; i++) {
        double w1 = rnd.drand32(-1.0, 1.0);
        double w2 = rnd.drand32(-1.0, 1.0);
        random_weights.push_back(vnl_vector_fixed<double, 2>(w1, w2));
    }
    // @todo not used in the program
    
    vector<PTZSplitParameter> candidate_splits;
    for (int i = 0; i<pixel_offset_candidate_num; i++) {
        int x = rand()%max_pixel_offset;
        int y = rand()%max_pixel_offset;
        int c1 = rand()%3;
        int c2 = rand()%3;
        
        PTZSplitParameter split_param;
        split_param.offset2_[0] = x;
        split_param.offset2_[1] = y;
        split_param.channle_[0] = c1;
        split_param.channle_[1] = c2;
        split_param.wt_[0] = 1.0;
        split_param.wt_[1] = -1.0;
        
        candidate_splits.push_back(split_param);
    }
    
    double loss = std::numeric_limits<double>::max();
    bool is_split = false;
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    PTZSplitParameter split_param;
    for (int i = 0; i<candidate_splits.size(); i++) {
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        double cur_split_value = 0.0;
        // optimize the offset
        double cur_loss = bestSplitCandidate(samples,
                                             rgb_images,
                                             indices,
                                             candidate_splits[i],
                                             split_candidate_num,
                                             min_leaf_node,
                                             cur_left_indices,
                                             cur_right_indices,
                                             cur_split_value);
        if (cur_loss < loss) {
            loss = cur_loss;
            is_split = true;
            left_indices  = cur_left_indices;
            right_indices = cur_right_indices;
            split_param = candidate_splits[i];
            split_param.threshold_ = cur_split_value;
        }
    }
    
    if (is_split) {
        if (param_.verbose_) {
            double ratio = 1.0*left_indices.size()/indices.size();
            printf("left right node size is %lu %lu, ratio is %lf\n", left_indices.size(), right_indices.size(), ratio);
        }
        assert(left_indices.size() + right_indices.size() == indices.size());
        node->is_leaf_ = false;
        node->depth_   = depth;
        node->sample_num_ = (int)indices.size();
        node->split_param_ = split_param;
        if (indices.size() < 500) {
            // statistics of pan, tilt and zoom
            PTZTreeUtil::mean_std(samples, indices, node->ptz_, node->stddev_);
            // statistics of sampled colors
            vector<vnl_vector_fixed<double, 3>> sample_colors;
            for (int i = 0; i<indices.size(); i++) {
                sample_colors.push_back(samples[indices[i]].color_);
            }
            PTZTreeUtil::mean_std(sample_colors, node->color_mu_, node->color_sigma_);
        }
        
        if (left_indices.size() > 0) {
            PTZTreeNode *left_node = new PTZTreeNode(depth + 1);
            assert(left_node);
            left_node->sample_percentage_ = 1.0 * left_indices.size()/indices.size();
            this->configureNode(samples, rgb_images, left_indices, depth + 1, left_node);
            node->left_child_ = left_node;
        }
        
        if (right_indices.size() > 0) {
            PTZTreeNode * right_node = new PTZTreeNode(depth + 1);
            assert(right_node);
            right_node->sample_percentage_ = 1.0 * right_indices.size()/indices.size();
            this->configureNode(samples, rgb_images, right_indices, depth + 1, right_node);
            node->right_child_ = right_node;
        }
    }
    else
    {
        node->depth_ = depth;
        node->is_leaf_ = true;
        node->sample_num_ = (int)indices.size();
        PTZTreeUtil::mean_std(samples, indices, node->ptz_, node->stddev_);
        vector<vnl_vector_fixed<double, 3>> sample_colors;
        for (int i = 0; i<indices.size(); i++) {
            sample_colors.push_back(samples[indices[i]].color_);
        }
        
        PTZTreeUtil::mean_std(sample_colors, node->color_mu_, node->color_sigma_);
        if (param_.verbose_) {
            //cout<<"pan tilt zoom is "<<node->ptz_<<endl;
            printf("depth, num_leaf_node, %d, %lu\n", depth, indices.size());
            cout<<"pan tilt zoom "<<node->ptz_<<endl;
            cout<<"pan tilt zoom standard deviation is "<<node->stddev_<<endl;
            cout<<"mean color "<<node->color_mu_<<endl;
            cout<<"color stddev "<<node->color_sigma_<<endl;
        }
        return true;
    }
    
    return true;
}


bool PTZTree::predict(
                      const vil_image_view<unsigned char> & rgb_image,
                      PTZTestingResult & predict) const
{
    assert(root_);
    return this->predict(root_, rgb_image, predict);
}

bool PTZTree::predict(PTZTreeNode *node,
                      const vil_image_view<unsigned char> & rgb_image,
                      PTZTestingResult & predict) const
{
    assert(node);
    
    if (node->is_leaf_) {        
        predict.predict_ptz_ = node->ptz_;
        predict.predict_color_ = node->color_mu_;
        return true;
    }
    
    PTZSplitParameter & split_param = node->split_param_;
    
    vnl_vector_fixed<int, 2> p1 = predict.img2d_;
    vnl_vector_fixed<int, 2> p2 = predict.img2d_ + split_param.offset2_;
    int v1 = rgb_image(p1[0], p1[1], split_param.channle_[0]);
    int v2 = 0;
    if (rgb_image.in_range(p2[0], p2[1])) {
        v2 = rgb_image(p2[0], p2[1], split_param.channle_[1]);
    }
    double feat = v1 * split_param.wt_[0] + v2 * split_param.wt_[1];
    if (feat < split_param.threshold_ && node->left_child_) {
        return this->predict(node->left_child_, rgb_image, predict);
    }
    else if(node->right_child_) {
        return this->predict(node->right_child_, rgb_image, predict);
    }
    else {
        return false;
    }
    
    return true;
}

/****************         PTZTreeNode             ******************/

static void write_PTZ_prediction(FILE *pf, PTZTreeNode * node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    // write current node
    PTZSplitParameter param = node->split_param_;
    fprintf(pf, "%2d %d %8d %3.2f %1d\t %3d %3d %1d\t %12.6f %12.6f %12.6f\t",
            node->depth_, (int)node->is_leaf_,  node->sample_num_, node->sample_percentage_, param.channle_[0],
            param.offset2_[0], param.offset2_[1], param.channle_[1],
            param.wt_[0], param.wt_[1], param.threshold_);
    
    fprintf(pf, "%8.3f %8.3f %8.3f\t %6.3f %6.3f %6.3f\t %6.1f %6.1f %6.1f\t %6.1f %6.1f %6.1f\n",
            node->ptz_[0], node->ptz_[1], node->ptz_[2],
            node->stddev_[0], node->stddev_[1], node->stddev_[2],
            node->color_mu_[0], node->color_mu_[1], node->color_mu_[2],
            node->color_sigma_[0], node->color_sigma_[1], node->color_sigma_[2]);
    
    write_PTZ_prediction(pf, node->left_child_);
    write_PTZ_prediction(pf, node->right_child_);
}


static void read_PTZ_prediction(FILE *pf, PTZTreeNode * & node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    if (lineBuf[0] == '#') {
        // empty node
        node = NULL;
        return;
    }
    
    // read node parameters
    node = new PTZTreeNode(0);
    assert(node);
    int depth = 0;
    int isLeaf = 0;
    int sample_num = 0;
    double sample_percentage = 0.0;
    
    int d2[2] = {0};
    int c1 = 0;        // rgb image channel
    int c2 = 0;
    double wt[2] = {0.0};
    double threhold = 0;
    double ptz[3] = {0.0};
    double sigma_ptz[3] = {0.0};
    double color[3] = {0.0};
    double color_sigma[3] = {0.0};
    
    int ret_num = sscanf(lineBuf, "%d %d %d %lf %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                         &depth, &isLeaf, & sample_num, &sample_percentage, &c1,
                         &d2[0], &d2[1], &c2,
                         &wt[0], &wt[1], &threhold,
                         &ptz[0], &ptz[1], &ptz[2],
                         &sigma_ptz[0], &sigma_ptz[1], &sigma_ptz[2],
                         &color[0], &color[1], &color[2],
                         &color_sigma[0], &color_sigma[1], &color_sigma[2]);
    
    assert(ret_num == 23);
    
    node->depth_ = depth;
    node->is_leaf_ = (isLeaf == 1);
    node->ptz_ = vnl_vector_fixed<double, 3>(ptz[0], ptz[1], ptz[2]);
    node->stddev_ = vnl_vector_fixed<double, 3>(sigma_ptz[0], sigma_ptz[1], sigma_ptz[2]);
    node->color_mu_ = vnl_vector_fixed<double, 3>(color);
    node->color_sigma_ = vnl_vector_fixed<double, 3>(color_sigma);
    node->sample_num_ = sample_num;
    node->sample_percentage_ = sample_percentage;
    
    // split parameter
    PTZSplitParameter param;
    param.offset2_ = vnl_vector_fixed<int, 2>(d2[0], d2[1]);
    param.channle_ = vnl_vector_fixed<int, 2>(c1, c2);
    param.wt_ = vnl_vector_fixed<double, 2>(wt);
    param.threshold_ = threhold;
    
    node->split_param_ = param;
    
    node->left_child_  = NULL;
    node->right_child_ = NULL;
    
    read_PTZ_prediction(pf, node->left_child_);
    read_PTZ_prediction(pf, node->right_child_);
}


bool PTZTreeNode::writeTree(const char *fileName, PTZTreeNode * root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t sample_num\t sample_percentage\t c1\t displace2\t c2\t w1 w2\t threshold\t ptz stddev_ptz \t mean_color stddev_color\n");
    
    write_PTZ_prediction(pf, root);
    fclose(pf);
    return true;
}

bool PTZTreeNode::readTree(const char *fileName, PTZTreeNode * &root)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    //read first line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
    read_PTZ_prediction(pf, root);
    fclose(pf);
    return true;
}


