//
//  main.cpp
//  Relocalization
//
//  Created by jimmy on 4/15/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include <iostream>
#include <string>
#include <vector>
#include "PTZRegressor.h"
#include "PTZRegressorBuilder.h"
#include "PTZTreeUtil.h"

#if 1


static void help()
{
    printf("program   sequenceParamFile  decisionTreeParameterFile  saveFile\n");
    printf("PTZ_train seq1_train.txt     RF_param.txt               rgb_RF.txt\n");
    printf("parameter fits to US high school basketball dataset\n");
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
    if (argc != 4) {
        printf("argc is %d, should be 4\n", argc);
        help();
        return -1;
    }
   
    const char * sequence_param  = argv[1];
    const char * tree_param_file = argv[2];
    const char * save_model_file = argv[3];
    
    string sequence_file_name;
    string image_sequence_base_dir;
    read_testing_sequence_files(sequence_param, sequence_file_name, image_sequence_base_dir);
    printf("sequence file is %s\n", sequence_file_name.c_str());
    printf("image sequence base directory is %s\n", image_sequence_base_dir.c_str());
    
    PTZTreeParameter tree_param;
    bool is_read = tree_param.readFromFile(tree_param_file);
    assert(is_read);
    tree_param.printSelf();
    if (tree_param.is_use_depth_) {
        printf("Note: depth is used ..................................................................\n");
    }
    else {
        printf("Note: depth is NOT used ..................................................................\n");
    }
    
    // read images and labels (ground truth)
    vector<string> image_files;
    vector<vnl_vector_fixed<double, 3> > ptzs;
    PTZTreeUtil::read_sequence_data(sequence_file_name.c_str(), image_sequence_base_dir.c_str(), image_files, ptzs);
    assert(image_files.size() == ptzs.size());
    printf("training frame number is %lu\n", image_files.size());
    
    PTZRegressorBuilder builder;
    builder.setTreeParameter(tree_param);
    
    PTZRegressor model;
    builder.buildModel(model, image_files, ptzs, save_model_file);
    model.save(save_model_file);
    return 0;
}

#endif


