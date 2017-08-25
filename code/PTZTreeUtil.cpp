//
//  PTZTreeUtil.cpp
//  Relocalization
//
//  Created by jimmy on 4/16/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "PTZTreeUtil.h"
#include <assert.h>
//#include "vpgl_plus.h"
//#include <vgl/vgl_point_2d.h>
#include <stdlib.h>
//#include <vnl/vnl_math.h>

vector<PTZLearningSample> PTZTreeUtil::generateLearningSamples(const Eigen::Vector3d &ptz,
                                                               const int img_width,
                                                               const int img_height,
                                                               const int img_index,
                                                               const int sample_num)
{
    vector<PTZLearningSample> samples;
    
    Eigen::Vector2d pp(img_width/2.0, img_height/2.0);
    for (int i = 0; i<sample_num; i++) {
        PTZLearningSample sp;
        
        int x = rand()%img_width;
        int y = rand()%img_height;
        Eigen::Vector2d p(x, y);
        
      
        PTZTreeUtil::panYtiltXFromPrinciplePointEncodeFocalLength(pp, ptz, p, sp.ptz_);
        
        sp.img2d_ = Eigen::Vector2i(x, y);
        sp.image_index_ = img_index;
        
        samples.push_back(sp);
    }
    
    return samples;
}

vector<PTZLearningSample>
PTZTreeUtil::generateLearningSamples(const Eigen::Tensor<unsigned char, 3> & image,
                                     const Eigen::Vector3d &ptz,
                                     const int img_index,
                                     const int sample_num,
                                     const Eigen::AlignedBox<double, 2> & exclusive_region)
{
    assert(image.dimension(2) == 3);
    const long img_height = image.dimension(0);
    const long img_width = image.dimension(1);
    
    vector<PTZLearningSample> samples;
    Eigen::Vector2d pp(img_width/2.0, img_height/2.0);
    for (int i = 0; i<sample_num; i++) {
        PTZLearningSample sp;
        
        int x = rand()%img_width;
        int y = rand()%img_height;
        
        Eigen::Vector2d p(x, y);
        if (exclusive_region.contains(p)) {
            // exclude this position
            continue;
        }
        double r = image(y, x, 0);
        double g = image(y, x, 1);
        double b = image(y, x, 2);
        
        PTZTreeUtil::panYtiltXFromPrinciplePointEncodeFocalLength(pp, ptz, p, sp.ptz_);
        
        sp.img2d_ = Eigen::Vector2i(x, y);
        sp.image_index_ = img_index;
        sp.color_ = Eigen::Vector3d(r, g, b);
        samples.push_back(sp);
    }
    return samples;
}



void PTZTreeUtil::mean_std(const vector<PTZLearningSample> & samples,
                           const vector<unsigned int> & indices,
                           Eigen::Vector3d & mean,
                           Eigen::Vector3d & stddev)
{
    assert(samples.size() > 0);
    
    mean.fill(0.0);
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        mean += samples[index].ptz_;
    }
    
    mean /= indices.size();
    
    stddev.fill(0.0);
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        Eigen::Vector3d dif = samples[index].ptz_ - mean;
        for (int j = 0; j<dif.size(); j++) {
            stddev[j] += dif[j] * dif[j];
        }
    }
    
    for (int i = 0; i<stddev.size(); i++) {
        stddev[i] = sqrt(stddev[i]/indices.size());
    }
}

/*
void PTZTreeUtil::mean_std(const vector<vnl_vector_fixed<double, 3> > & data,
                           vnl_vector_fixed<double, 3> & mean,
                           vnl_vector_fixed<double, 3> & stddev)
{
    assert(data.size() > 0);
    
    mean.fill(0.0);
    for (int i = 0; i<data.size(); i++) {
        mean += data[i];
    }
    mean /= data.size();
    
    stddev.fill(0.0);
    for (int i = 0; i<data.size(); i++) {
        vnl_vector_fixed<double, 3> dif = data[i] - mean;
        for (int j = 0; j<dif.size(); j++) {
            stddev[j] += dif[j] * dif[j];
        }
    }
    
    for (int i = 0; i<stddev.size(); i++) {
        stddev[i] = sqrt(stddev[i]/data.size());
    }
}
*/

double PTZTreeUtil::variance(const vector<PTZLearningSample> & samples,
                             const vector<unsigned int> & indices,
                             const Eigen::Vector3d & wt)
{
    assert(samples.size() > 0);
    
    Eigen::Vector3d avg(0, 0, 0);
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        avg += samples[index].ptz_;
    }
    
    avg /= indices.size();
    
    Eigen::Vector3d variance(0, 0, 0);
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        Eigen::Vector3d dif = samples[index].ptz_ - avg;
        for (int j = 0; j<dif.size(); j++) {
            variance[j] += dif[j] * dif[j];
        }
    }
    
    double ret = 0.0;
    for (int i = 0; i<wt.size(); i++) {
        ret += wt[i] * variance[i];
    }
    return ret;
}
/*
void PTZTreeUtil::median_error(const vector< vnl_vector_fixed<double, 3> > & error,
                               vnl_vector_fixed<double, 3> & median)
{
    assert(error.size() > 0);
    
    vector<vector<double> > values(3);
    for (int i = 0; i<error.size(); i++) {
        for (int j = 0; j<error[i].size(); j++) {
            values[j].push_back(fabs(error[i][j]));
        }
    }
    for (int i = 0; i<3; i++) {
        std::sort(values[i].begin(), values[i].end());
        double m = values[i][values[i].size()/2];
        median[i] = m;
    }
}
*/

void PTZTreeUtil::read_sequence_data(const char * sequence_file_name,
                                 const char * image_sequence_base_directory,
                                 vector<string> & image_files,
                                 vector<Eigen::Vector3d > & ptzs)
{    
    assert(sequence_file_name);
    assert(image_sequence_base_directory);
    
    FILE *pf = fopen(sequence_file_name, "r");
    if (!pf) {
        printf("can not open: %s\n", sequence_file_name);
        return;
    }
    // skip three rows;
    for (int i = 0; i<3; i++) {
        char buf[1024] = {NULL};
        fgets(buf, sizeof(buf), pf);
        printf("%s\n", buf);
    }
    while (1) {
        char buf[1024] = {NULL};
        Eigen::Vector3d ptz;
        int ret = fscanf(pf, "%s %lf %lf %lf", buf, &ptz[0], &ptz[1], &ptz[2]);// @ not sure, if it works
        if (ret != 4) {
            break;
        }
        image_files.push_back(image_sequence_base_directory + string(buf));
        ptzs.push_back(ptz);
    }
    assert(ptzs.size() == image_files.size());
    fclose(pf);
    printf("load %lu files\n", image_files.size());
}

bool PTZTreeUtil::load_prediction_result_with_color(const char *file_name,
                                       string & rgb_img_file,
                                       Eigen::Vector3d & ptz,
                                       vector<Eigen::Vector2i > & img_pts,
                                       vector<Vector3d > & pred_ptz,
                                       vector<Vector3d > & gt_ptz,
                                       vector<Vector3d > & color_pred,
                                       vector<Vector3d > & color_sample)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error, can not read from %s\n", file_name);
        return false;
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s\n", buf); // remove the last \n
        rgb_img_file = string(buf);
    }
    
    {
        char dummy_buf[1024] = {NULL};
        fgets(dummy_buf, sizeof(dummy_buf), pf);
        //printf("%s\n", dummy_buf);
    }
    
    {
        double data[3] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf", &data[0], &data[1], &data[2]);
        assert(ret == 3);
        for (int i =0; i<3; i++) {
            ptz[i] = data[i];
        }
    }
    while (1) {
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }
        
        // 2D , 3D position
        img_pts.push_back(Eigen::Vector2i(val[0], val[1]));
        pred_ptz.push_back(Eigen::Vector3d(val[2], val[3], val[4]));
        gt_ptz.push_back(Eigen::Vector3d(val[5], val[6], val[7]));
        
        double val2[6] = {0.0};
        ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf",
                     &val2[0], &val2[1], &val2[2],
                     &val2[3], &val2[4], &val2[5]);
        if (ret != 6) {
            break;
        }
        color_pred.push_back(Eigen::Vector3d(val2[0], val2[1], val2[2]));
        color_sample.push_back(Eigen::Vector3d(val2[3], val2[4], val2[5]));
        assert(img_pts.size() == color_pred.size());
    }
    fclose(pf);
    printf("read %lu prediction and ground truth points.\n", img_pts.size());
    return true;
}

void PTZTreeUtil::panTiltFromReferencePointDecodeFocalLength(const Eigen::Vector2d & reference_point,
                                                          const Eigen::Vector3d & reference_ptz,
                                                          const Eigen::Vector2d & principle_point,
                                                          Eigen::Vector3d & ptz)
{
    double dx = reference_point.x() - principle_point.x();
    double dy = reference_point.y() - principle_point.y();
    double ref_pan  = reference_ptz[0];
    double ref_tilt = reference_ptz[1];
    double fl = reference_ptz[2];
    
    double delta_pan  = atan2(dx, fl) * 180.0/M_PI;
    double delta_tilt = atan2(dy, fl) * 180.0/M_PI;
    
    ptz[0] = ref_pan  - delta_pan;
    ptz[1] = ref_tilt + delta_tilt;
    if (fl*fl - dx*dx - dy*dy > 0) {
        ptz[2] = sqrt(fl*fl - dx*dx - dy*dy);
    }
    else  {
        ptz[2] = fl;
    }
}

void PTZTreeUtil::panYtiltXFromPrinciplePointEncodeFocalLength(const Eigen::Vector2d & pp,
                                                            const Eigen::Vector3d & pp_ptz,
                                                               const Eigen::Vector2d & p2,
                                                            Vector3d & p2_ptz)
{
    double dx = p2.x() - pp.x();
    double dy = p2.y() - pp.y();
    
    double pan_pp  = pp_ptz[0];
    double tilt_pp = pp_ptz[1];
    double focal_length = pp_ptz[2];
    double delta_pan  = atan2(dx, focal_length) * 180/M_PI;
    double delta_tilt = atan2(dy, focal_length) * 180/M_PI;
    
    double pan2  = pan_pp + delta_pan;
    double tilt2 = tilt_pp - delta_tilt;  // y is upside down
    p2_ptz[0] = pan2;
    p2_ptz[1] = tilt2;
    p2_ptz[2] = sqrt(focal_length * focal_length + dx * dx + dy * dy);
}



