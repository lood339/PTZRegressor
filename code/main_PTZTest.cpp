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
#include <vil/vil_load.h>
#include <iostream>
#include <vgl/vgl_point_2d.h>
//#include "vpgl_plus.h"

using std::cout;
using std::endl;



static void help()
{
    printf("program  RFModelFile sequenceParam numSample average saveFilePrefix \n");
    printf("PTZ_test rf.txt      seq_param.txt 5000      0       ptz_predict_   \n");
    printf("parameter fits to US high school basketball dataset\n");
    printf("average: 0 --> best color, 1 --> average from all trees\n");
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
    if (argc != 6) {
        printf("argc is %d, should be 6\n", argc);
        help();
        return -1;
    }
    const char * model_file = argv[1];
    const char * sequence_param = argv[2];
    int num_sample = strtod(argv[3], NULL);
    bool is_average = (strtod(argv[4], NULL) == 1);
    const char * save_file_prefix = argv[5];
    
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
    vector<vnl_vector_fixed<double, 3> > ptzs;
    PTZTreeUtil::read_sequence_data(sequence_file_name.c_str(), image_sequence_base_dir.c_str(), image_files, ptzs);
    
    
    vgl_box_2d<double> invalid_region = PTZTreeUtil::highschool_RS_invalid_region();
    cout<<"invalid region is "<<invalid_region<<endl;
    
    for (int i = 0; i<image_files.size(); i++) {
        
        // sample from the image
        vil_image_view<vxl_byte> cur_img = vil_load(image_files[i].c_str());
        vnl_vector_fixed<double, 3> cur_ptz = ptzs[i];
        assert(cur_img.nplanes()  == 3);
        
        
        vector<PTZLearningSample> cur_samples;
        cur_samples = PTZTreeUtil::generateLearningSamples(cur_img, cur_ptz, i, num_sample, invalid_region);
        //printf("sampler number is %lu\n", cur_samples.size());
        
        // predict each pixel PTZ
        double tt = clock();
        vgl_point_2d<double> pp(cur_img.ni()/2.0, cur_img.nj()/2.0);
        vector<PTZTestingResult> predictions;
        vector<vnl_vector_fixed<double, 3> > prediction_errors;
        for (int j = 0; j<cur_samples.size(); j++) {
            PTZTestingResult tr;
            tr.img2d_ = cur_samples[j].img2d_;
            tr.sampled_color_ = cur_samples[j].color_;
            bool is_predict = false;
            if (is_average) {
                is_predict = model.predictAverage(cur_img, tr);
            }
            else
            {
                is_predict = model.predictByColor(cur_img, tr);
            }
            
            tr.gt_ptz_ = cur_samples[j].ptz_;
            if (is_predict) {
                vnl_vector_fixed<double, 3> estimated_ptz;
                vgl_point_2d<double> ref_pt(tr.img2d_[0], tr.img2d_[1]);
                PTZTreeUtil::panTiltFromReferencePointDecodeFocalLength(ref_pt, tr.predict_ptz_, pp, estimated_ptz);
                vnl_vector_fixed<double, 3> err = cur_ptz - estimated_ptz;
                
                predictions.push_back(tr);
                prediction_errors.push_back(err);
            }
        }
        printf("RF prediction cost time %lf\n", (clock() - tt)/CLOCKS_PER_SEC);
        
        
        
        // median error
        vnl_vector_fixed<double, 3> median_error;
        PTZTreeUtil::median_error(prediction_errors, median_error);
        cout<<"prediction median error is "<<median_error<<endl;
        
        // write to file
        {
            char save_file[1024] = {NULL};
            sprintf(save_file, "%s_%06d.txt", save_file_prefix, i);
            FILE *pf = fopen(save_file, "w");
            assert(pf);
            fprintf(pf, "%s\n", image_files[i].c_str());
            fprintf(pf, "imageLocation\t  prediction3d groundTruth3d \t predicted_color actual_color\n");
            fprintf(pf, "%lf %lf %lf\n", cur_ptz[0], cur_ptz[1], cur_ptz[2]);
            for (int j = 0; j<predictions.size(); j++) {
                vnl_vector_fixed<int, 2> p1 = predictions[j].img2d_;
                vnl_vector_fixed<double, 3> p2 = predictions[j].predict_ptz_;
                vnl_vector_fixed<double, 3> p3 = predictions[j].gt_ptz_;
                vnl_vector_fixed<double, 3> color_1 = predictions[j].predict_color_;
                vnl_vector_fixed<double, 3> color_2 = predictions[j].sampled_color_;
                
                fprintf(pf, "%6d %6d\t %lf %lf %lf\t %lf %lf %lf\t", p1[0], p1[1], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]);
                fprintf(pf, "%4.1f %4.1f %4.1f\t %4.1f %4.1f %4.1f\n", color_1[0], color_1[1], color_1[2], color_2[0], color_2[1], color_2[2]);
            }
            fclose(pf);
            printf("save to %s\n", save_file);
        }
    }
    printf("test from %lu frames\n", image_files.size());
    return 0;
}

#endif


