//
// Created by niw on 2019/6/28.
//

#ifndef DECISIONTREE_LOAD_DATA_H
#define DECISIONTREE_LOAD_DATA_H
#include <iostream>
#include <string.h>
#include <vector>
using namespace std;

struct Problem{
    vector<vector<float >> X;
    vector<int> y;
    uint32_t feature_size;
    uint32_t data_cnt;
    uint32_t num_classes;
};
bool LoadData(Problem& p, string path);

#endif //DECISIONTREE_LOAD_DATA_H
