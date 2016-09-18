//
//  PTZ_tree.h
//  Relocalization
//
//  Created by jimmy on 4/16/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Relocalization__PTZ_tree__
#define __Relocalization__PTZ_tree__

#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_vector.h>
#include <vil/vil_image_view.h>
#include <vector>
#include "PTZTreeUtil.h"

using std::vector;

class PTZTreeNode;

class PTZTree
{
    friend class PTZRegressor;
    PTZTreeNode *root_;
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
                   const vector<vil_image_view<unsigned char> > & rgb_images,
                   const PTZTreeParameter & param);
    
    bool predict(const vil_image_view<unsigned char> & rgb_image,
                 PTZTestingResult & predict) const;
    

    
private:
    bool configureNode(const vector<PTZLearningSample> & samples,
                       const vector<vil_image_view<unsigned char> > & rgb_images,
                       const vector<unsigned int> & indices,
                       int depth,
                       PTZTreeNode *node);
    
    bool predict(PTZTreeNode *node,                 
                 const vil_image_view<unsigned char> & rgb_image,
                 PTZTestingResult & predict) const;


    
    
};

struct PTZSplitParameter
{
    vnl_vector_fixed<int, 2> offset2_;        // displacement in image, [x, y]
    vnl_vector_fixed<int, 2> channle_;        // rgb image channel
    vnl_vector_fixed<double, 2> wt_;
    double threshold_;  // threshold of splitting. store result
    
    PTZSplitParameter()
    {
        offset2_ = vnl_vector_fixed<int, 2>(0, 0);
        channle_ = vnl_vector_fixed<int, 2>(0, 0);
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
    vnl_vector_fixed<double, 3> ptz_;
    vnl_vector_fixed<double, 3> stddev_;  // standard deviation of the samples
    int sample_num_;
    double sample_percentage_;
    
    vnl_vector_fixed<double, 3> color_mu_;
    vnl_vector_fixed<double, 3> color_sigma_;
    
    PTZTreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
        sample_num_ = 0;
        sample_percentage_ = 0.0;
        ptz_    = vnl_vector_fixed<double, 3>(0, 0, 0);
        stddev_ = vnl_vector_fixed<double, 3>(0, 0, 0);
        color_mu_ = vnl_vector_fixed<double, 3>(0, 0, 0);
        color_sigma_ = vnl_vector_fixed<double, 3>(0, 0, 0);
    }
    
    static bool writeTree(const char *fileName, PTZTreeNode * root);
    static bool readTree(const char *fileName, PTZTreeNode * &root);
};

#endif /* defined(__Relocalization__PTZ_tree__) */
