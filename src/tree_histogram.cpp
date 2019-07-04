//
// Created by niw on 2019/7/4.
//
#include "tree_histogram.h"

void TreeLeafWiseLearner::fit(Bin& bin, Problem& prob)
{
    vector<int> train_data_inds = random.Sample(data_size, rowsample*data_size);
    vector<int> train_feature_inds = random.Sample(feature_size, colsample*feature_size);
    build(bin, prob, train_data_inds, train_feature_inds);
}

bool TreeLeafWiseLearner::build(Bin& bin, Problem& prob, vector<int>& train_data_inds, vector<int>& train_feature_inds){
    root = new TreeLeafWiseNode;
    split_candidate_nodes.push_back(root);
    leaf_node_indices.resize(train_data_inds.size());
    memcpy(leaf_node_indices.data(), train_data_inds.data(), sizeof(int)*train_data_inds.size());
    root->set_leaf_node_index_from(0);
    root->set_leaf_node_index_to(leaf_node_indices.size());
    tree_depth = 1;
    while(tree_depth<=max_depth){
        int best_split_node_index = -1;
        int best_feature_index = -1;
        int best_bin_index = -1;
        float best_gain = 1.0;
        //遍历当前所有的叶子节点
        for (int node_index = 0; node_index < split_candidate_nodes.size(); ++node_index) {
            //遍历所有的feature
            TreeLeafWiseNode* leaf_node = split_candidate_nodes[node_index];
            if(leaf_node->has_child()) continue;
            for (int i = 0; i < train_feature_inds.size(); ++i) {
                int feature_index = train_feature_inds[i];
                //构建当前feature的直方图
                Histogram H = Histogram(bin.get_num_bins(feature_index), num_classes);
                //通过当前叶子节点的数据计算当前feature的最佳分裂点
                int from = leaf_node->get_leaf_node_index_from();
                int to = leaf_node->get_leaf_node_index_to();
                for (int j = from; j <= to ; ++j) {
                    int data_index = leaf_node_indices[j];
                    int bin_index = bin.get_bin_index(data_index, feature_index);
                    H.add_sample(bin_index, prob.y[data_index]);
                }
                H.find_best_split();
                float gain = H.get_best_gini();
                int bin_index = H.get_best_split();
                if(gain < best_gain){
                    best_gain = gain;
                    best_bin_index = bin_index;
                    best_feature_index = feature_index;
                    best_split_node_index = node_index;
                }
            }
        }
        //split
        assert(best_split_node_index >=0);
        TreeLeafWiseNode* best_split_node = split_candidate_nodes[best_split_node_index];
        vector<int> left_data_indices;
        vector<int> right_data_indices;
        vector<float> left_probs;
        vector<float> right_probs;
        left_probs.resize(num_classes);
        right_probs.resize(num_classes);
        int from = best_split_node->get_leaf_node_index_from();
        int to = best_split_node->get_leaf_node_index_to();
        for (int j = from; j <= to ; ++j) {
            int data_index = leaf_node_indices[j];
            int bin_index = bin.get_bin_index(data_index, best_feature_index);
            if(bin_index<=best_bin_index){
                left_data_indices.push_back(data_index);
                left_probs[prob.y[data_index]] += 1.0;
            }else{
                right_data_indices.push_back(data_index);
                right_probs[prob.y[data_index]] += 1.0;
            }
        }
        for (int class_index = 0; class_index < num_classes; ++class_index) {
            left_probs[class_index]/=left_data_indices.size();
            right_probs[class_index]/=right_data_indices.size();
        }
        int left_child_from = from;
        int right_child_from = from+left_data_indices.size();
        memcpy(leaf_node_indices.data()+from, left_data_indices.data(), sizeof(int)*left_data_indices.size());
        memcpy(leaf_node_indices.data()+right_child_from, right_data_indices.data(), sizeof(int)*right_data_indices.size());
        TreeLeafWiseNode* left_node = new TreeLeafWiseNode;
        left_node->set_leaf_node_index_from(left_child_from);
        left_node->set_leaf_node_index_to(right_child_from-1);
        left_node->set_probs(left_probs);

        TreeLeafWiseNode* right_node = new TreeLeafWiseNode;
        right_node->set_leaf_node_index_from(right_child_from);
        right_node->set_leaf_node_index_to(to);
        right_node->set_probs(right_probs);

        best_split_node->set_left_child(left_node);
        best_split_node->set_right_child(right_node);

        if(left_node->get_sample_cnt() > 10 && left_node->get_gini() > 0.2){
            split_candidate_nodes.push_back(left_node);
        }

        if(right_node->get_sample_cnt() > 10 && right_node->get_gini() > 0.2){
            split_candidate_nodes.push_back(right_node);
        }
        tree_depth++;
    }
    for (int k = 0; k < split_candidate_nodes.size(); ++k) {
        if(!split_candidate_nodes[k]->has_child()){
            leaf_nodes_cnt++;
        }
        nodes_cnt++;
    }
    split_candidate_nodes.clear();

    return true;
}