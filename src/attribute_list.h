//
// Created by niw on 2019/6/28.
//

#ifndef DECISIONTREE_ATTRIBUTE_LIST_H
#define DECISIONTREE_ATTRIBUTE_LIST_H

#include "load_data.h"
#include <assert.h>
#include <map>
#include <algorithm>

class Bin{
public:
    Bin(){};
    static bool comp_by_value(const pair<float, int> &p1, const pair<float, int> &p2){
        return p1.first < p2.first;
    }
    bool build(Problem& prob){
        vector<vector<float >>& X = prob.X;
        bins.resize(prob.feature_size);
        data_feature_bin_index.resize(prob.data_cnt);
        #pragma omp parallel for schedule(dynamic)
        for (int feature_index = 0; feature_index< prob.feature_size; feature_index++){
            map<float, int> distinct_values_ind_cnt;

            for (int data_index = 0; data_index < prob.data_cnt; ++data_index) {
                float val = X[data_index][feature_index];
                /*if (distinct_values_ind_cnt.find(val) == distinct_values_ind_cnt.end()){
                    distinct_values_ind_cnt[val]=0;
                }*/
                distinct_values_ind_cnt[val]+=1;
            }
            vector<pair<float, int> > distinct_value_inds_vec (distinct_values_ind_cnt.begin(), distinct_values_ind_cnt.end());
            sort(distinct_value_inds_vec.begin(),distinct_value_inds_vec.end(),comp_by_value);
            uint32_t distinct_value_cnt = distinct_value_inds_vec.size();
            if(distinct_value_cnt<=max_bins){
                for (int i = 0; i < distinct_value_inds_vec.size(); ++i) {
                    bins[feature_index].push_back(distinct_value_inds_vec[i].first);
                }

            }else{
                int avg_cnt = prob.data_cnt/max_bins;
                int acc_cnt = 0;
                for (int i = 0; i < distinct_value_cnt; ++i) {
                    acc_cnt += distinct_value_inds_vec[i].second;
                    if (acc_cnt >= avg_cnt){
                        //cout<<"feature_index:"<<feature_index<<" adding:"<<distinct_value_inds_vec[i].first<<endl;
                        bins[feature_index].push_back(distinct_value_inds_vec[i].first);
                        acc_cnt=0;
                    }
                }
                if(acc_cnt != 0){
                    bins[feature_index].push_back(distinct_value_inds_vec[distinct_value_cnt-1].first);
                }
            }
            bins[feature_index].push_back(numeric_limits<float>::infinity());
            //cout<<bins[feature_index].size()<<endl;
            //break;

        }
        #pragma omp parallel for schedule(dynamic)
        for (int data_index = 0; data_index < prob.data_cnt; ++data_index) {
            data_feature_bin_index[data_index].resize(prob.feature_size);
            for (int feature_index = 0; feature_index < prob.feature_size; ++feature_index) {
                float val = X[data_index][feature_index];
                int index = this->find_bin_index(feature_index, val);
                data_feature_bin_index[data_index][feature_index] = index;
            }
        }

        return true;
    }

    inline int find_bin_index(int feature_index, float feature_value){
        vector<float>& bin_upper_bound = bins[feature_index];

        for (int bin_index = 0; bin_index < bin_upper_bound.size(); ++bin_index) {
            if(feature_value<= bin_upper_bound[bin_index]){
                return bin_index;
            }
        }
        return -1;
    }

    inline int get_num_bins(int feature_index){
        return bins[feature_index].size();
    }
    inline int get_bin_index(int data_index, int feature_index){
        return data_feature_bin_index[data_index][feature_index];
    }

    vector<vector<int>> discrete_data(vector<vector<float >>& X){
        uint64_t data_size = X.size();
        uint64_t feature_size = X[0].size();
        vector<vector<int>> res;
        res.resize(data_size);
        for (uint64_t data_index = 0; data_index < data_size; ++data_index) {
            res[data_index].resize(feature_size);
            for (uint64_t feature_index = 0; feature_index < feature_size; ++feature_index) {
                float val = X[data_index][feature_index];
                int bin_index = find_bin_index(feature_index, val);
                res[data_index][feature_index]=bin_index;
            }
        }
        return res;
    }

    void clean_up(){
        bins.clear();
    }

    int get_max_bins(){return max_bins+1;}
private:
    uint16_t max_bins=255;
    vector<vector<float >>bins;
    //为每个feature建立一个bin，每个bin中存储了对特征值进行分桶的阈值，最多256个(max_bin+1),最后一个值为inf
    //此结构用于对连续特征值进行分桶
    //分桶的好处：1.原本数据要存储浮点型的特征值，分桶后只需存储bin的index，减少存储
    //          2.决策树在尝试对某个特征做切分时，原本需要遍历distinct_vaules_cnt次，分桶后只需max_bin次


    //建桶的空间复杂度 #feature_size * max_bins
    //建桶的时间复杂度 #feature_size * (#data_cnt)log(#data_cnt)
    vector<vector<int>> data_feature_bin_index;

};

class AttributeList{
public:
    AttributeList(){}
    ~AttributeList(){}
    bool build(Bin& bin, Problem& prob){
        vector<vector<float >>& X = prob.X;
        feature_bin_list.resize(prob.feature_size);
        data_feature_bin_index.resize(prob.data_cnt);
        //遍历每个样本下的每个feature
        //按照分桶上界值，对样本进行分桶
        //同一个样本找到对应的#feature_size个桶
        //复杂度:#data_size * #feature_size * #max_bins
        for (int data_index = 0; data_index < prob.data_cnt; ++data_index) {
            data_feature_bin_index[data_index].resize(prob.feature_size);
            for (int feature_index = 0; feature_index < prob.feature_size; ++feature_index) {
                //vector<float>& bin_upper_bound = bins[feature_index];
                float val = X[data_index][feature_index];
                int bin_index = bin.get_bin_index(feature_index, val);
                assert(bin_index>=0);
                //bin_index start from zero
                //omp_set_lock(&writelock);
                if(feature_bin_list[feature_index].size()<bin_index+1){
                    feature_bin_list[feature_index].resize(bin_index+1);
                }
                //omp_unset_lock(&writelock);
                feature_bin_list[feature_index][bin_index].push_back(data_index);
                data_feature_bin_index[data_index][feature_index]=bin_index;

            }
        }
        return true;

    }

    inline int get_bin_index(int data_index, int feature_index){
        return data_feature_bin_index[data_index][feature_index];
    }

    void clean_up(){
        feature_bin_list.clear();
        data_feature_bin_index.clear();
    }

    vector<vector<vector<int>>> feature_bin_list;
    //#feature_size * #bin_size * #data_index_in_bin
    vector<vector<int>> data_feature_bin_index;
    //#data_size * #feature_size
};
#endif //DECISIONTREE_ATTRIBUTE_LIST_H
