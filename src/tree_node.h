//
// Created by niw on 2019/6/28.
//

#ifndef DECISIONTREE_TREE_NODE_H
#define DECISIONTREE_TREE_NODE_H

#include <vector>
#include <assert.h>
using namespace std;
class TreeNode{
public:
    TreeNode(int num_classes){
        this->num_classes = num_classes;
        best_split_gini = 1.0;
        is_leaf = false;
        total_sample_cnt = 0;
        max_prob = -1.0;
        left_child = nullptr;
        right_child = nullptr;
        clear_up();

    }

    int get_best_split_feature_index(){return best_split_feature_index;}
    int get_best_split_upper_bounder_index(){return best_split_upper_bounder_index;}

    inline void acc_left(int class_label){
        left_histogram[class_label]++;
    }

    void clear_left(){
        for (int i = 0; i < num_classes; ++i) {
            left_histogram[i] = 0;
        }

    }
    void clear_up(){

        left_histogram.clear();
        right_histogram.clear();
        total_histogram.clear();
        left_histogram.resize(num_classes);
        right_histogram.resize(num_classes);
        total_histogram.resize(num_classes);

    }

    inline void calc_gini(int feature_index, int upper_bounder_index){
        int left_cnt = 0;
        int right_cnt = 0;
        for (int i = 0; i < num_classes; ++i) {
            right_histogram[i]=total_histogram[i]-left_histogram[i];
            if(right_histogram[i]<0){
                assert(0);
            }
            left_cnt += left_histogram[i];
            right_cnt += right_histogram[i];
        }
        float left_gini=0.0;
        float right_gini = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            float left_prob_i = static_cast<float>(left_histogram[i])/static_cast<float>(left_cnt);
            left_gini += left_prob_i*(1-left_prob_i);
            float right_prob_i = static_cast<float>(right_histogram[i])/static_cast<float>(right_cnt);
            right_gini += right_prob_i*(1-right_prob_i);

        }

        float left_prob = static_cast<float>(left_cnt)/static_cast<float>(left_cnt+right_cnt);
        float right_prob = 1-left_prob;
        float gini = left_prob*left_gini+right_prob*right_gini;
        if(gini<best_split_gini){
            best_split_gini=gini;
            best_split_feature_index = feature_index;
            best_split_upper_bounder_index = upper_bounder_index;
        }

    }

    inline void acc_total(int class_label){
        total_histogram[class_label]++;
        total_sample_cnt ++;
    }

    int get_sample_cnt(){
        return total_sample_cnt;
    }

    bool contain_one_class(){
        for (int i = 0; i < total_histogram.size(); ++i) {
            if(total_histogram[i]==0)
                return true;
        }
        return false;
    }

    inline bool is_leaf_node(){
        return is_leaf;
    }

    void set_leaf(bool is_or_not = true){
        is_leaf= is_or_not;
        if(is_or_not){
            probs.resize(num_classes);
            for (int i = 0; i < num_classes; ++i) {
                float prob = static_cast<float>(total_histogram[i])/static_cast<float>(total_sample_cnt);
                probs[i] = prob;
                if(prob > max_prob){
                    max_prob = prob;
                    max_prob_class = i;
                }

            }
        }
    }

    vector<float> get_class_probs(){
        return probs;
    }

    int get_class(){ return max_prob_class;}
    float get_class_prob(){return max_prob;}

    TreeNode* get_left_child(){
        return left_child;
    }

    TreeNode* get_right_child(){
        return right_child;
    }

    void set_left_child(TreeNode* node){left_child = node;}
    void set_right_child(TreeNode* node){right_child = node;}
    int num_classes;
    float best_split_gini;
    int best_split_feature_index;
    int best_split_upper_bounder_index;
    bool is_leaf;
    int total_sample_cnt;
    vector<int> left_histogram;
    vector<int> right_histogram;
    vector<int> total_histogram;
    vector<float> probs;
    int max_prob_class;
    float max_prob;
    TreeNode* left_child;
    TreeNode* right_child;
};
#endif //DECISIONTREE_TREE_NODE_H
