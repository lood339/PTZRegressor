//
//  PTZRegressor.h
//  Relocalization
//
//  Created by jimmy on 5/4/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Relocalization__PTZRegressor__
#define __Relocalization__PTZRegressor__

#include "PTZ_tree.h"
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

using std::vector;

class PTZRegressor
{
    friend class PTZRegressorBuilder;
    
    vector<PTZTree *> trees_;
    
public:
    PTZRegressor();
    ~PTZRegressor();
    
    // average value from all decision trees
    bool predictAverage(const Eigen::Tensor<unsigned char, 3> & rgb_image,
                        PTZTestingResult & predict) const;
    
    // result from one tree whose leaf node has the most similar color as sampler
    bool predictByColor(const Eigen::Tensor<unsigned char, 3> & rgb_image,
                        PTZTestingResult & predict) const;
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
    
};



#endif /* defined(__Relocalization__PTZRegressor__) */
