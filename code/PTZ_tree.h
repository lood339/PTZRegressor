//
//  PTZ_tree.h
//  Relocalization
//
//  Created by jimmy on 4/16/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Relocalization__PTZ_tree__
#define __Relocalization__PTZ_tree__


#include <Eigen/Dense>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include "PTZTreeUtil.h"
#include "dt_random.hpp"

using std::vector;
using Eigen::Vector3d;
using Eigen::Vector2d;

class PTZTreeNode;
struct PTZSplitParameter;

class PTZTree
{
    friend class PTZRegressor;
    PTZTreeNode *root_;
    DTRandom rnd_generator_;
    PTZTreeParameter param_;
    
public:
    PTZTree()
    {
        root_ = NULL;
    }
    
    ~PTZTree(){}
    
    PTZTreeNode *rootNode()
    {
        return root_;
    }
    
    void setRootNode(PTZTreeNode * node)
    {
        root_ = node;
    }
    
    void setTreeParameter(const PTZTreeParameter & param)
    {
        param_ = param;
    }

    
    // training random forest by build a decision tree
    // samples: sampled image pixel locations
    // indices: index of samples
    // rgbImages: same size, rgb, 8bit image
    bool buildTree(const vector<PTZLearningSample> & samples,
                   const vector<unsigned int> & indices,
                   const vector<Eigen::Tensor<unsigned char, 3 > > & rgb_images,  // h x w x 3
                   const PTZTreeParameter & param);
    
    bool predict(const Eigen::Tensor<unsigned char, 3 > & rgb_image,
                 PTZTestingResult & predict) const;
    

    
private:
    bool configureNode(const vector<PTZLearningSample> & samples,
                       const vector<Eigen::Tensor<unsigned char, 3 > >  & rgb_images,
                       const vector<unsigned int> & indices,
                       int depth,
                       PTZTreeNode *node);
    
    bool predict(PTZTreeNode *node,                 
                 const Eigen::Tensor<unsigned char, 3 > & rgb_image,
                 PTZTestingResult & predict) const;
    
    double bestSplitCandidate(const vector<PTZLearningSample> & samples,
                              const vector<Eigen::Tensor<unsigned char, 3 > > & rgb_images,
                              const vector<unsigned int> & indices,
                              const PTZSplitParameter & split_param,
                              const int split_num,
                              const int min_leaf_node,
                              vector<unsigned int> & left_indices,
                              vector<unsigned int> & right_indices,
                              double & split_value);


    
    
};

struct PTZSplitParameter
{
    Eigen::Vector2i offset2_;        // displacement in image, [x, y]
    Eigen::Vector2i channle_;        // rgb image channel
    Eigen::Vector2d wt_;
    double threshold_;  // threshold of splitting. store result
    
    PTZSplitParameter()
    {
        offset2_.setConstant(0.0);
        channle_.setConstant(0.0);
        wt_[0] = 1.0;
        wt_[1] = -1.0;
        threshold_ = 0.0;
    }
};

class PTZTreeNode
{
public:
    PTZTreeNode *left_child_;
    PTZTreeNode *right_child_;
    int depth_;
    bool is_leaf_;
    
    PTZSplitParameter split_param_;
    // for leaf node
    Vector3d ptz_;
    Vector3d stddev_;  // standard deviation of the samples
    int sample_num_;
    double sample_percentage_;
    
    Vector3d color_mu_;
    Vector3d color_sigma_;
    
    PTZTreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
        sample_num_ = 0;
        sample_percentage_ = 0.0;
        ptz_.setConstant(0.0);
        stddev_.setConstant(0.0);
        color_mu_.setConstant(0.0);
        color_sigma_.setConstant(0.0);
    }
    
    static bool writeTree(const char *fileName, PTZTreeNode * root);
    static bool readTree(const char *fileName, PTZTreeNode * &root);
};

#endif /* defined(__Relocalization__PTZ_tree__) */
