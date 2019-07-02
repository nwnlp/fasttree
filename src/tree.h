//
// Created by niw on 2019/6/28.
//

#ifndef DECISIONTREE_TREE_H
#define DECISIONTREE_TREE_H
#include "attribute_list.h"
#include "tree_node.h"
#include "class_list.h"
#include <set>
#include "tools.h"
class Tree{
public:
    Tree(int id, int max_depth, float colsample, float rowsample){
        this->max_depth = max_depth;
        this->colsample = colsample;
        this->rowsample = rowsample;
        tree_depth = 0;
        attribute_list = nullptr;
        tree_id = id;
    }

    ~Tree(){
        clean_up();
    }

    void try_split(int feature_index, int upper_bounder_index, vector<int>& left_data){
        for (int i = 0; i < left_data.size(); ++i) {
            int data_index = left_data[i];
            TreeNode* node = class_list.get_node_by_index(data_index);
            if(node == nullptr || node->is_leaf_node())
                continue;
            node->acc_left(class_list.get_label_by_index(data_index));
        }
        //当前要分裂的节点计算增益
        for (int j = 0; j < split_candidate_nodes.size(); ++j) {
            TreeNode* node = split_candidate_nodes[j];
            node->calc_gini(feature_index, upper_bounder_index);
        }

    }

    void split(vector<int>& train_data_inds){
        for (int i = 0; i < train_data_inds.size(); ++i) {
            int data_index = train_data_inds[i];
            TreeNode* node = class_list.get_node_by_index(data_index);
            assert(node);
            if(node->is_leaf_node()){
                continue;
            }
            //数据所在节点最佳分割点
            int split_feature_index = node->get_best_split_feature_index();
            int split_upper_bounder_index = node->get_best_split_upper_bounder_index();
            int data_bin_index = attribute_list->get_bin_index(data_index, split_feature_index);
            if(data_bin_index<=split_upper_bounder_index){
                TreeNode* left_child = node->left_child;
                assert(left_child != nullptr);
                //对数据重新分配节点
                class_list.setTreeNode(data_index, left_child);
                //对新的节点累积计算直方图
                left_child->acc_total(class_list.get_label_by_index(data_index));
            } else{
                TreeNode* right_child = node->right_child;
                assert(right_child != nullptr);
                class_list.setTreeNode(data_index, right_child);
                right_child->acc_total(class_list.get_label_by_index(data_index));
            }
        }

    }


    void  check_leaf_node(vector<TreeNode* > child_nodes, int level, int min_sample_in_node = 1){
        split_candidate_nodes.clear();
        for (int i = 0; i < child_nodes.size(); ++i) {
            TreeNode* node = child_nodes[i];
            if(node == nullptr){
                continue;
            }
            if(node->get_sample_cnt()<=min_sample_in_node
               || level == max_depth
               || node->contain_one_class()){
                //if(node->contain_one_class()){
                //cout<<"node name:"<<node->get_name()<<" contain one class"<<endl;
                //tt += node->get_sample_cnt();
                //cout<<tt<<endl;
                //}
                node->set_leaf(true);
                leaf_node_cnt += 1;
            } else{
                //need split nodes in current level
                split_candidate_nodes.push_back(node);
                node->set_leaf(false);

            }

        }
    }


    void clear_node_left_histogram(){
        for (int i = 0; i < split_candidate_nodes.size(); ++i) {
            split_candidate_nodes[i]->clear_left();
        }
    }

    void split_eval(vector<int>& train_feature_inds){
        vector<vector<vector<int>>>& feature_bin_list = attribute_list->feature_bin_list;
        for (int i = 0; i < train_feature_inds.size(); ++i) {
            int feature_index = train_feature_inds[i];
            vector<vector<int>> &upper_bounder = feature_bin_list[feature_index];
            for (int upper_bounder_index = 0; upper_bounder_index < upper_bounder.size(); ++upper_bounder_index) {
                vector<int> &left_data = upper_bounder[upper_bounder_index];
                try_split(feature_index, upper_bounder_index,left_data);

            }
            //单个feature的左直方图可以做累加，换feature的时候要清零
            clear_node_left_histogram();
        }
    }

    vector<TreeNode* > new_child_for_candidates(){
        vector<TreeNode* > new_nodes;
        for (int i = 0; i < split_candidate_nodes.size(); ++i) {
            TreeNode* left = new TreeNode(num_classes);
            split_candidate_nodes[i]->set_left_child(left);

            TreeNode* right = new TreeNode(num_classes);
            split_candidate_nodes[i]->set_right_child(right);
            new_nodes.push_back(left);
            new_nodes.push_back(right);
            node_cnt += 2;
        }
        return new_nodes;
    }

    bool build(vector<int>& train_data_inds, vector<int>& train_feature_inds){

        root = new TreeNode(num_classes);
        //每个样本的初始节点为根节点
        class_list.initTreeNode(root, train_data_inds);
        //为根节点建立直方图
        class_list.buildHistogram();
        node_cnt = 1;
        leaf_node_cnt = 0;
        //TODO colsample here
        if(!root->contain_one_class())
            split_candidate_nodes.push_back(root);
        int depth = 1;
        while (depth<=max_depth) {
            //cout<<"split_candidate_nodes size:"<<split_candidate_nodes.size()<<endl;
            if(split_candidate_nodes.empty()){
                break;
            }
            vector<TreeNode* > child_nodes = new_child_for_candidates();
            split_eval(train_feature_inds);
            //cout<<"split eval done depth:"<<depth<<endl;
            split(train_data_inds);
            //cout<<"split done depth:"<<depth<<endl;
            check_leaf_node(child_nodes, depth);
            tree_depth = depth;
            depth++;
        }
        return true;

    }

    void delete_node(TreeNode* node){
        TreeNode* left = node->left_child;
        TreeNode* right = node->right_child;
        delete node;
        if(left) delete_node(left);
        if(right) delete_node(right);

    }
    void clean_up(){
        delete_node(root);
    }

    void fit(AttributeList* attribute_list_, vector<int>& y){
        //cout<<"start fitting tree:"<< tree_id<<endl;
        attribute_list = attribute_list_;
        num_classes = attribute_list->get_num_classes();
        data_size = attribute_list->get_data_size();
        feature_size = attribute_list->get_feature_size();
        Random random = Random(tree_id);
        vector<int> train_data_inds = random.Sample(data_size, rowsample*data_size);
        vector<int> train_feature_inds = random.Sample(feature_size, colsample*feature_size);
        //cout<<"sampler done"<<endl;
        //cout<<"train_data_size:"<<train_data_inds.size()<<" feature_size:"<<train_feature_inds.size()<<endl;
        class_list.build(train_data_inds, y,attribute_list->get_num_classes());
        //cout<<"build class_list done..."<<endl;
        build(train_data_inds, train_feature_inds);
        //cout<<"build tree done\t tree depth:"<<tree_depth<<" tree nodes:"<<node_cnt<<" leaf nodes:"<<leaf_node_cnt<<endl;
        class_list.clean_up();
    }

    vector<float> predict_one(vector<int>& x){
        TreeNode* node = root;
        while(!node->is_leaf_node()){
            int feature_index = node->get_best_split_feature_index();
            int upper_bounder_index = node->get_best_split_upper_bounder_index();
            if(x[feature_index]<=upper_bounder_index){
                node = node->left_child;
            } else{
                node = node->right_child;
            }
        }
        //found leaf node
        //node->get_class_probs();
        return node->get_class_probs();


    }

    vector<vector<float>>  predict_prob(vector<vector<int>>& data){
        vector<vector<float>> preds;
        for (int data_index = 0; data_index < data.size(); ++data_index) {
            vector<float> pred = predict_one(data[data_index]);
            preds.push_back(pred);
        }
        return preds;
    }

private:
    int tree_id;
    int max_depth;
    int tree_depth;
    float colsample;
    float rowsample;
    int data_size;
    int feature_size;
    int num_classes;
    TreeNode* root;
    int node_cnt;
    int leaf_node_cnt;
    vector<TreeNode* > split_candidate_nodes;
    ClassList class_list;
    AttributeList* attribute_list;

};
#endif //DECISIONTREE_TREE_H
