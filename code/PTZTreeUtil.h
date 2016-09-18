//
//  PTZTreeUtil.h
//  Relocalization
//
//  Created by jimmy on 4/16/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Relocalization__PTZTreeUtil__
#define __Relocalization__PTZTreeUtil__

#include <vnl/vnl_vector_fixed.h>
#include <vector>
#include <vnl/vnl_vector.h>
#include <unordered_map>
#include <vgl/vgl_box_2d.h>
#include <vil/vil_image_view.h>

using std::vector;
using std::unordered_map;
using std::string;

class PTZLearningSample
{
public:
    vnl_vector_fixed<int, 2>    img2d_;  // 2d location (x, y)
    vnl_vector_fixed<double, 3> ptz_;    // PTZ parameter in world coordinate, label
    int image_index_;           // image index
    vnl_vector_fixed<double, 3> color_;
};

class PTZTestingResult
{
public:
    vnl_vector_fixed<int, 2>    img2d_;   // image position
    vnl_vector_fixed<double, 3> gt_ptz_;  // as ground truth, not used in prediction
    vnl_vector_fixed<double, 3> predict_ptz_;   // predicted world coordinate
    
    vnl_vector_fixed<double, 3> std_;    // prediction standard deviation
    vnl_vector_fixed<double, 3> sampled_color_;   // image color
    vnl_vector_fixed<double, 3> predict_color_;   // mean color from leaf node
    
    vnl_vector_fixed<double, 3> prediction_error()
    {
        return predict_ptz_ - gt_ptz_;
    }   
    
};

struct PTZTreeParameter
{
    bool is_average_;             // true --> average all trees as the result
    bool is_use_depth_;           // true --> use depth, false depth is constant 1.0
    int max_frame_num_;           // sampled frames for a tree
    int sampler_num_per_frame_;   // sampler numbers in one frame
    
    int tree_num_;                // number of trees
    int max_depth_;
    int min_leaf_node_;
    
    int max_pixel_offset_;            // in pixel
    int pixel_offset_candidate_num_;  // large number less randomness
    int split_candidate_num_;  // number of split in [v_min, v_max]
    int weight_candidate_num_;
    bool verbose_;
    
    PTZTreeParameter()
    {
        // sampler parameters
        is_average_   = true;
        is_use_depth_ = false;
        max_frame_num_ = 500;
        sampler_num_per_frame_ = 5000;
        
        // tree structure parameter
        tree_num_ = 5;
        max_depth_ = 15;
        min_leaf_node_ = 1;
        
        // random sample parameter
        max_pixel_offset_ = 131;
        pixel_offset_candidate_num_ = 20;
        split_candidate_num_ = 20;
        weight_candidate_num_ = 0;
        verbose_ = true;
    }
    
    bool readFromFile(FILE *pf)
    {
        assert(pf);
        
        const int param_num = 12;
        unordered_map<std::string, int> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            int val = 0;
            int ret = fscanf(pf, "%s %d", s, &val);
            if (ret != 2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 12);
        
        is_average_ = (imap[string("is_average")] == 1);
        is_use_depth_ = (imap[string("is_use_depth")] == 1);
        max_frame_num_ = imap[string("max_frame_num")];
        sampler_num_per_frame_ = imap[string("sampler_num_per_frame")];
        
        tree_num_ = imap[string("tree_num")];
        max_depth_ = imap[string("max_depth")];
        min_leaf_node_ = imap[string("min_leaf_node")];
        
        max_pixel_offset_ = imap[string("max_pixel_offset")];
        pixel_offset_candidate_num_ = imap[string("pixel_offset_candidate_num")];
        split_candidate_num_ = imap[string("split_candidate_num")];
        
        weight_candidate_num_ = imap[string("weight_candidate_num")];
        verbose_ = imap[string("verbose")];
        
        return true;
    }
    
    bool readFromFile(const char *file_name)
    {
        assert(file_name);
        FILE *pf = fopen(file_name, "r");
        if(!pf){
            printf("can not open %s\n", file_name);
            return false;
        }
        this->readFromFile(pf);
        fclose(pf);
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "is_average %d\n", is_average_);
        fprintf(pf, "is_use_depth %d\n", is_use_depth_);
        fprintf(pf, "max_frame_num %d\n", max_frame_num_);
        fprintf(pf, "sampler_num_per_frame %d\n", sampler_num_per_frame_);
        
        fprintf(pf, "tree_num %d\n", tree_num_);
        fprintf(pf, "max_depth %d\n", max_depth_);
        fprintf(pf, "min_leaf_node %d\n", min_leaf_node_);
        
        fprintf(pf, "max_pixel_offset %d\n", max_pixel_offset_);
        fprintf(pf, "pixel_offset_candidate_num %d\n", pixel_offset_candidate_num_);
        fprintf(pf, "split_candidate_num %d\n", split_candidate_num_);
        fprintf(pf, "weight_candidate_num %d\n", weight_candidate_num_);
        fprintf(pf, "verbose %d\n", (int)verbose_);
        return true;
    }
    
    void printSelf() const
    {
        printf("RGB tree parameters:\n");
        printf("is_average: %d\t is_use_depth: %d\t max_frame_num: %d\n", is_average_, is_use_depth_, max_frame_num_);
        printf("tree_num: %d\t max_depth: %d\t min_leaf_node: %d\n", tree_num_, max_depth_, min_leaf_node_);
        printf("max_pixel_offset: %d\t pixel_offset_candidate_num: %d\t split_candidate_num %d\n",
               max_pixel_offset_,
               pixel_offset_candidate_num_,
               split_candidate_num_);
        printf("weight_candidate_num_: %d\n\n", weight_candidate_num_);
    }
};

class PTZTreeUtil
{
public:
    
    static vector<PTZLearningSample> generateLearningSamples(const vnl_vector_fixed<double, 3> &ptz,
                                                             const int img_width,
                                                             const int img_height,
                                                             const int img_index,
                                                             const int sample_num);
    //  exclusive_region: not not sample in this area
    static vector<PTZLearningSample>
           generateLearningSamples(const vil_image_view<vxl_byte> & image,
                                   const vnl_vector_fixed<double, 3> &ptz,
                                   const int img_index,
                                   const int sample_num,
                                   const vgl_box_2d<double> & exclusive_region);
    
    static void mean_std(const vector<PTZLearningSample> & samples,
                         const vector<unsigned int> & indices,
                         vnl_vector_fixed<double, 3> & mean,
                         vnl_vector_fixed<double, 3> & stddev);
    
    static void mean_std(const vector<vnl_vector_fixed<double, 3> > & data,
                         vnl_vector_fixed<double, 3> & mean,
                         vnl_vector_fixed<double, 3> & stddev);
    
    static double variance(const vector<PTZLearningSample> & samples,
                           const vector<unsigned int> & indices,
                           const vnl_vector_fixed<double, 3> & wt);
    
    // median value of absolute error
    static void median_error(const vector< vnl_vector_fixed<double, 3> > & error,
                             vnl_vector_fixed<double, 3> & median);
    
    //sequence_file_name            = "/Users/jimmy/Desktop/images/Youtube/PTZRegressor/seq1/seq1.txt"
    //image_sequence_base_directory = "/Users/jimmy/Desktop/images/Youtube/PTZRegressor/seq1/"
    static void read_sequence_data(const char * sequence_file_name,
                                   const char * image_sequence_base_directory,
                                   vector<string> & image_files,
                                   vector<vnl_vector_fixed<double, 3> > & ptzs);
    
    // load predicted results
    static bool load_prediction_result_with_color(const char *file_name,
                                           string & rgb_img_file,
                                           vnl_vector_fixed<double, 3> & ptz,
                                           vector<vnl_vector_fixed<int, 2> > & img_pts,
                                           vector<vnl_vector_fixed<double, 3> > & pred_ptz,
                                           vector<vnl_vector_fixed<double, 3> > & gt_ptz,
                                           vector<vnl_vector_fixed<double, 3> > & color_pred,
                                           vector<vnl_vector_fixed<double, 3> > & color_sample);
    
    // two points can be from different image
    static void panTiltFromReferencePointDecodeFocalLength(const vgl_point_2d<double> & reference_point,
                                                              const vnl_vector_fixed<double, 3> & reference_ptz,
                                                              const vgl_point_2d<double> & principle_point,
                                                           vnl_vector_fixed<double, 3> & ptz);
    
    // two points can be from different image
    // pp: pinciple point
    // pp_ptz: pan, tilt and zoom of the pinciple point
    // p2: a query point
    // p2_ptz: pan, tilt and zoom of the query point    
    static void panYtiltXFromPrinciplePointEncodeFocalLength(const vgl_point_2d<double> & pp,
                                                                   const vnl_vector_fixed<double, 3> & pp_ptz,
                                                                   const vgl_point_2d<double> & p2,
                                                             vnl_vector_fixed<double, 3> & p2_ptz);


    
    
    
    
    static inline bool is_inside_image(const int x, const int y, const int width, const int height)
    {
        return x >= 0 && x < width && y >= 0 && y < height;
    }
    
    static vgl_box_2d<double> highschool_RS_invalid_region()
    {
        vgl_box_2d<double> box(172.0, 172.0 + 936.0, 620.0,  620.0 + 66.0);
        return box;
    }
};



#endif /* defined(__Relocalization__PTZTreeUtil__) */
