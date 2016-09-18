//
//  PTZRegressorBuilder.h
//  Relocalization
//
//  Created by jimmy on 5/4/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Relocalization__PTZRegressorBuilder__
#define __Relocalization__PTZRegressorBuilder__

#include "PTZRegressor.h"
#include <vector>
#include <string>


using std::vector;
using std::string;

class PTZRegressorBuilder
{
    PTZTreeParameter tree_param_;
public:
    PTZRegressorBuilder(){}
    ~PTZRegressorBuilder(){}
    
    void setTreeParameter(const PTZTreeParameter & param)
    {
        tree_param_ = param;
    }
    
    //Build model from data
    bool buildModel(PTZRegressor& model,
                    const vector<PTZLearningSample> & samples,
                    const vector<vil_image_view<vxl_byte> > & rgbImages,
                    const char *model_file_name) const;
    
    // build model from subset of images without dropout    
    bool buildModel(PTZRegressor& model,
                    const vector<string> & rgb_img_files,
                    const vector<vnl_vector_fixed<double, 3> > & ptzs,
                    const char *model_file_name) const;
    
    //: Name of the class
    std::string is_a() const {return std::string("PTZRegressorBuilder");};
    
public:
    static void outof_bag_sampling(const unsigned int N,
                                   vector<unsigned int> & bootstrapped,
                                   vector<unsigned int> & outof_bag);
private:
    //
    bool validation_error(const PTZRegressor & model,
                          const vector<string> & rgb_img_files,
                          const vector<vnl_vector_fixed<double, 3> > & ptzs,
                          const int sample_frame_num = 100) const;
    
    
};

#endif /* defined(__Relocalization__PTZRegressorBuilder__) */
