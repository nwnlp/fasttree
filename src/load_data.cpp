//
// Created by niw on 2019/6/28.
//
#include "load_data.h"
#include <set>
bool LoadData(Problem& p, string path){
    FILE *f = fopen(path.c_str(), "r");
    if(!f) {
        std::cout << std::string("cannot open ") + path << std::endl;
        return false;
    }
    uint32_t nr_instance = 0;
    uint32_t const kMaxLineSize = 1000000;
    char line[kMaxLineSize];
    //ignore first line
    fgets(line, kMaxLineSize, f);
    char value[128]={0};
    set<int> distinct_labels;
    for(; fgets(line, kMaxLineSize, f) != nullptr; ++nr_instance)
    {
        int start = 0;int end = 0;
        vector<float> x;
        int label;
        for (int i = 0; i < kMaxLineSize; ++i) {
            if(line[i] == '\n'){
                strncpy(value, &line[start], end-start+1);
                label = atoi(value);
                //cout<<label<<endl;
                break;
            }
            if(line[end]==','){
                float f_value;
                if(end == start){
                    //缺失的值用-999填充
                    f_value = -999.0;
                }
                else{
                    strncpy(value, &line[start], end-start+1);
                    value[end+1]=0;
                    f_value = atof(value);
                }
                //cout<<f_value<<endl;
                x.push_back(f_value);
                start = end+1;
            }
            end++;
        }
        p.X.push_back(x);
        p.y.push_back(label);
        distinct_labels.insert(label);
    }
    p.feature_size = p.X[0].size();
    p.data_cnt = p.X.size();
    p.num_classes=distinct_labels.size();
    return true;
}

