//
//  main.cpp
//  Relocalization
//
//  Created by jimmy on 4/15/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 1
#include <iostream>
#include <string>
#include <vector>
#include "PTZRegressor.h"
#include "PTZRegressorBuilder.h"
#include "PTZTreeUtil.h"
#include <iostream>
#include "cvx_tensor.h"
#include "dt_util.hpp"
#include "ptz_pose_estimation.h"
#include "mat_io.hpp"
#include "eigenVLFeatSIFT.h"
#include "cvxImgMatch.h"
#include "cvxImage_310.hpp"
#include "pgl_ptz_camera.h"

using std::cout;
using std::endl;
using Eigen::Vector2d;
using Eigen::Vector3d;


static void help()
{
    printf("program  databaseSequenceParam testSequenceParam initialPTZ numSample saveFile \n");
    printf("PTZ_test leave_1_out.txt       seq1.txt          seq1_result.mat      seq1_refine_result.mat   \n");
    printf("PTZ camera pose estimation using SIFT matching and two-point pose optimization\n");
    printf("parameter fits to US high school basketball dataset\n");
    printf("principal point is fixed at (1280/2, 720/2) \n");
}

bool readSequenceFiles(const char *file_name,
                         string & sequence_name,
                         string & image_sequence_base_dir);

bool readImagenamesPtz(const char *file_name, vector<string> & image_names, vector<Eigen::Vector3d > & gt_ptzs);

bool pointMtching(const char * image_name1, const char * image_nam2,
                  vector<Vector2d> & pts1, vector<Eigen::Vector2d> & pts2);

int main(int argc, const char * argv[])
{
    if (argc != 5) {
        printf("argc is %d, should be 5 \n", argc);
        help();
        return -1;
    }
    
    const char * database_seq_param = argv[1]; // database sequence ptz
    const char * test_sequence_param = argv[2];
    const char * estimated_ptz_file = argv[3];
    const char * save_file = argv[4];
    
    /*
    const char * database_seq_param = "/Users/jimmy/Desktop/learn_program/bmvc16_ptz_calib/data/seq1_train.txt"; // database sequence ptz
    const char * test_sequence_param = "/Users/jimmy/Desktop/learn_program/bmvc16_ptz_calib/data/seq1_test.txt";
    const char * estimated_ptz_file = "/Users/jimmy/Desktop/learn_program/bmvc16_ptz_calib/analyze_result/seq1_result.mat";
    const char * save_file = "temp.mat";
     */
    
    // Step 1: read database ptz and image_names
    vector<string> database_image_files;
    vector<Eigen::Vector3d > database_ptzs;
    bool is_read = readImagenamesPtz(database_seq_param, database_image_files, database_ptzs);
    assert(database_image_files.size() == database_ptzs.size());
    
    // read testing image names and estimated ptz
    vector<string> test_image_names;
    vector<Eigen::Vector3d> test_ground_truth_ptz;
    is_read = readImagenamesPtz(test_sequence_param, test_image_names, test_ground_truth_ptz);
    assert(test_image_names.size() == test_ground_truth_ptz.size());
    
    Eigen::MatrixXd init_ptzs;
    Eigen::MatrixXd gt_ptzs;
    is_read = matio::readMatrix(estimated_ptz_file, "estimated_ptz", init_ptzs);
    assert(is_read);
    is_read = matio::readMatrix(estimated_ptz_file, "ground_truth_ptz", gt_ptzs);
    assert(is_read);
    assert(init_ptzs.rows() == test_image_names.size());
    assert(init_ptzs.cols() == 3);
    
    Eigen::Vector2d pp(1280/2.0, 720/2.0);
    ptz_pose_opt::PTZPreemptiveRANSACParameter param;
    param.reprojection_error_threshold_ = 2.0;
    param.sample_number_ = 16;
    printf("principal point is fixed in (%lf %lf)\n", pp.x(), pp.y());
    
    // Step 2: optimize camera pose
    Eigen::MatrixXd optimized_ptzs = init_ptzs;  // default ptz is the initial ptz
    for(auto r = 0; r<init_ptzs.rows(); r++) {
        Eigen::Vector3d init_ptz = init_ptzs.row(r);
        // find nearest neighbor in the database
        double min_angle = INT_MAX;
        int min_index = -1;
        for (int i = 0; i<database_ptzs.size(); i++) {
            // use pan/tilt angle and L2 norm
            double dif_pan  = init_ptz[0] - database_ptzs[i][0];
            double dif_tilt = init_ptz[1] - database_ptzs[i][1];
            double dif = dif_pan * dif_pan + dif_tilt * dif_tilt;
            
            if(dif < min_angle) {
                min_angle = dif;
                min_index = i;
            }
        }
        assert(min_index != -1);
        
        double tt = clock();
        vector<Vector2d> src_pts;
        vector<Vector2d> dst_pts;
        pointMtching(database_image_files[min_index].c_str(), test_image_names[r].c_str(), src_pts, dst_pts);
        printf("inlier poitns after homography ransac is %lu\n", src_pts.size());
        if (src_pts.size() <= 4) {
            continue;
        }
                
        vector<Eigen::Vector2d> image_points = dst_pts;
        vector<vector<Eigen::Vector2d> > candidate_pan_tilt;
        Vector3d ref_ptz = database_ptzs[r];
        Vector3d est_ptz;
        
        for(int i = 0; i<src_pts.size(); i++) {
            vector<Eigen::Vector2d> pan_tilt_groups;
            pan_tilt_groups.push_back(cvx_pgl::point2PanTilt(pp, ref_ptz, src_pts[i]));
            candidate_pan_tilt.push_back(pan_tilt_groups);
        }
        assert(image_points.size() == candidate_pan_tilt.size());
        
        bool is_estimated = ptz_pose_opt::preemptiveRANSACOneToMany(image_points, candidate_pan_tilt, pp, param, est_ptz, false);
        if (is_estimated) {
            optimized_ptzs(r, 0) = est_ptz[0];
            optimized_ptzs(r, 1) = est_ptz[1];
            optimized_ptzs(r, 2) = est_ptz[2];
        }
        
        printf("Camera pose refinement cost time: %f seconds.\n", (clock() - tt)/CLOCKS_PER_SEC);
    }
    
    std::vector<std::string> var_name;
    std::vector<Eigen::MatrixXd> data;
    var_name.push_back("ground_truth_ptz");
    var_name.push_back("initial_ptz");
    var_name.push_back("optimized_ptz");
    data.push_back(gt_ptzs);
    data.push_back(init_ptzs);
    data.push_back(optimized_ptzs);
    matio::writeMultipleMatrix(save_file, var_name, data);
    
    
    vector<Eigen::VectorXd> errors;
    for (int r = 0; r<gt_ptzs.rows(); r++) {
        Eigen::VectorXd err = optimized_ptzs.row(r) - gt_ptzs.row(r);
        errors.push_back(err);
    }
    
    Eigen::VectorXd mean_error, median_error;
    DTUtil::meanMedianError(errors, mean_error, median_error);
    cout<<"PTZ mean   error: "<<mean_error.transpose()<<endl;
    cout<<"PTZ median error: "<<median_error.transpose()<<endl;
 
    return 0;
}


bool readSequenceFiles(const char *file_name,
                                        string & sequence_name,
                                        string & image_sequence_base_dir)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not open %s\n", file_name);
        return false;
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
    return true;
}

bool readImagenamesPtz(const char *file_name, vector<string> & image_names, vector<Eigen::Vector3d > & gt_ptzs)
{
    string seq_file_name;
    string seq_base_dir;
    bool is_read = readSequenceFiles(file_name, seq_file_name, seq_base_dir);
    assert(is_read);
   
    PTZTreeUtil::read_sequence_data(seq_file_name.c_str(), seq_base_dir.c_str(), image_names, gt_ptzs);
    assert(image_names.size() == gt_ptzs.size());
    return true;
}

bool pointMtching(const char * image_name1, const char * image_name2,
                  vector<Vector2d> & pts1, vector<Eigen::Vector2d> & pts2)
{
    cv::Mat im1 = cv::imread(image_name1);
    cv::Mat im2 = cv::imread(image_name2);
    
    assert(!im1.empty());
    assert(!im2.empty());
    
    vector<std::shared_ptr<sift_keypoint> > src_keypoints;
    vector<std::shared_ptr<sift_keypoint> > dst_keypoints;
    
    vl_feat_sift_parameter sift_para;
    sift_para.edge_thresh = 20;
    sift_para.peak_thresh = 1.0;
    sift_para.magnif      = 3.0;
    sift_para.dim = 128;
    EigenVLFeatSIFT::extractSIFTKeypoint(im1, sift_para, src_keypoints, true);
    EigenVLFeatSIFT::extractSIFTKeypoint(im2, sift_para, dst_keypoints, true);
    
    // SIFT matching and ransac filter
    vector<cv::Point2d> src_pts;
    vector<cv::Point2d> dst_pts;
    SIFTMatchingParameter matching_param;
    CvxImgMatch::SIFTMatching(src_keypoints, dst_keypoints, matching_param, src_pts, dst_pts);
    
    Mat mask;
    Mat H = cv::findHomography(src_pts, dst_pts, mask, CV_RANSAC, 3.0);
    assert(mask.rows == src_pts.size());
    
    int n_inlier = 0;
    for (int k = 0; k<src_pts.size(); k++) {
        if ((int)(mask.at<uchar>(k, 0)) == 1) {
            pts1.push_back(Vector2d(src_pts[k].x, src_pts[k].y));
            pts2.push_back(Vector2d(dst_pts[k].x, dst_pts[k].y));
            n_inlier++;
        }
    }
    assert(pts1.size() == pts2.size());
    return pts1.size() >= 4;
}



#endif


