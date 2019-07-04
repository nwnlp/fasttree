//
// Created by niw on 2019/7/4.
//

#ifndef RF_TREE_HISTOGRAM_H
#define RF_TREE_HISTOGRAM_H

#include <vector>
#include "tools.h"
#include "attribute_list.h"
using namespace std;
class TreeLeafWiseNode{
public:
    inline int get_leaf_node_index_from(){
        return leaf_node_index_from;
    }
    inline int get_leaf_node_index_to(){
        return leaf_node_index_to;
    }

    void set_leaf_node_index_from(int from){
        leaf_node_index_from = from;
    }
    void set_leaf_node_index_to(int to){
        leaf_node_index_to = to;
    }

    TreeLeafWiseNode* get_left_child(){
        return left_child;
    }

    TreeLeafWiseNode* get_right_child(){
        return right_child;
    }

    void set_left_child(TreeLeafWiseNode* node){
        left_child = node;
    }

    int get_sample_cnt(){
        return leaf_node_index_to-leaf_node_index_from+1;
    }

    void set_probs(vector<float>& probs){
        node_probs = probs;
        float gini = 0.0;
        for (int i = 0; i < probs.size(); ++i) {
            float p = probs[i];
            gini += p*(1-p);
        }
        this->gini = gini;
    }

    float get_gini(){
        return gini;
    }
    void set_right_child(TreeLeafWiseNode* node){right_child = node;}
    bool has_child(){return !(left_child== nullptr);}

private:
    int leaf_node_index_from;
    int leaf_node_index_to;
    TreeLeafWiseNode* left_child = nullptr;
    TreeLeafWiseNode* right_child = nullptr;
    vector<float> node_probs;
    float gini;
};

class Histogram{
public:
    Histogram(int num_bins, int num_classes){
        histogram.resize(num_bins);
        for (int bin_index = 0; bin_index < num_bins; ++bin_index) {
            histogram[bin_index].resize(num_classes);
        }
        class_histogram.resize(num_classes);
        this->num_classes = num_classes;
        this->num_bins = num_bins;
        total_cnt = 0;
    }

    void add_sample(int bin_index, int class_label){
        histogram[bin_index][class_label]+=1;
        class_histogram[class_label]+=1;
        total_cnt += 1;
    }

    int get_best_split(){
        return best_split_index;
    }

    float get_best_gini(){
        return best_gini;
    }
    void find_best_split(){
        vector<int> left_acc;
        left_acc.resize(num_classes);
        int left_cnt = 0;
        for (int bin_index = 0; bin_index < num_bins; ++bin_index) {
            for (int class_index = 0; class_index < num_classes; ++class_index) {
                left_acc[class_index] += histogram[bin_index][class_index];
                left_cnt += histogram[bin_index][class_index];
            }
            float left_gini = 0.0;
            float right_gini = 0.0;
            for (int class_index = 0; class_index < num_classes; ++class_index) {
                float left_prob_i = static_cast<float>(left_acc[class_index])/static_cast<float>(left_cnt);
                float right_prob_i = static_cast<float>(class_histogram[class_index]-left_acc[class_index])/
                        static_cast<float>(total_cnt-left_cnt);
                left_gini += left_prob_i*(1-left_prob_i);
                right_gini += right_prob_i*(1-right_prob_i);
            }
            float left_prob = static_cast<float>(left_cnt)/static_cast<float>(total_cnt);
            float right_prob = 1-left_prob;
            float gini = left_prob*left_gini+right_prob*right_gini;
            if(gini<best_gini){
                best_gini = gini;
                best_split_index = bin_index;
            }
        }
    }
private:
    vector<vector<int>> histogram;
    vector<int> class_histogram;
    int num_classes;
    int num_bins;
    int total_cnt;
    int best_split_index = -1;
    float best_gini=1.0;
};
class TreeLeafWiseLearner{
public:
    TreeLeafWiseLearner(int id, int max_depth, int data_size, int feature_size,int num_classes, float colsample, float rowsample){
        this->max_depth = max_depth;
        this->colsample = colsample;
        this->rowsample = rowsample;
        this->data_size = data_size;
        this->feature_size = feature_size;
        this->num_classes = num_classes;
        tree_depth = 0;
        tree_id = id;
    }
    ~TreeLeafWiseLearner(){}

    void fit(Bin& bin,Problem& prob);

    bool build(Bin& bin,Problem& prob, vector<int>& train_data_inds, vector<int>& train_feature_inds);

    int get_nodes_cnt(){return nodes_cnt;}
    int get_leaf_nodes_cnt(){return leaf_nodes_cnt;}
    int get_tree_depth(){return tree_depth;}

private:
    int tree_id;
    int max_depth;
    int tree_depth;
    float colsample;
    float rowsample;
    int data_size;
    int feature_size;
    int num_classes;
    vector<TreeLeafWiseNode*> split_candidate_nodes;
    vector<int> leaf_node_indices;
    TreeLeafWiseNode* root;
    Random random = Random(tree_id);
    int nodes_cnt = 0;
    int leaf_nodes_cnt = 0;
};
#endif //RF_TREE_HISTOGRAM_H
