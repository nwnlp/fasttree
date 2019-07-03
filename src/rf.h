//
// Created by johnny on 2019/7/1.
//

#ifndef DECISIONTREE_RF_H
#define DECISIONTREE_RF_H

#include <vector>
#include "tree.h"
#include "attribute_list.h"
#include <ctime>
#include <omp.h>
#include <unistd.h>
class RandomForest{

public:
    RandomForest(int num_trees, int max_depth, float colsample, float rowsample){
        this->num_trees = num_trees;
        this->max_depth = max_depth;
        this->colsample = colsample;
        this->rowsample = rowsample;

    }
    ~RandomForest(){
        for (int i = 0; i < forest.size(); ++i) {
            delete forest[i];
        }
        bin.clean_up();
    }
    int get_tree_nums(){
        return forest.size();
    }
    void fit(Problem& prob){
        bin.build(prob);
        cout<<"build bin done..."<<endl;
        attribute_list.build(bin, prob);
        num_classes = prob.num_classes;
        int data_size = prob.data_cnt;
        int feature_size = prob.feature_size;
        forest.resize(num_trees);
        //omp_set_num_threads(8);
        #pragma omp parallel for schedule(dynamic)
        for (int tree_id = 0; tree_id <num_trees; ++tree_id) {
            Tree* tree = new Tree(tree_id,max_depth,data_size, feature_size,num_classes, colsample,rowsample);
            tree->fit(&attribute_list, &(prob.y));
            int node_cnt = tree->get_node_cnt();
            int leaf_node_cnt = tree->get_leaf_node_cnt();
            cout<<"tree id:"<<tree_id<<" node count:"<<node_cnt<<" leaf node count:"<<leaf_node_cnt<<endl;
            forest[tree_id] = tree;
        }
        attribute_list.clean_up();
    }

    vector<int> predict(Problem& prob){
        vector<vector<int>> data = bin.discrete_data(prob.X);
        vector<int> y_pred;
        y_pred.resize(data.size());
        #pragma omp parallel for schedule(dynamic)
        for (int data_index = 0; data_index < data.size(); ++data_index) {
            vector<float> avg_probs;
            avg_probs.resize(num_classes);
            int label = -1;
            float max_prob= -numeric_limits<float>::infinity();
            for (int tree_id = 0; tree_id <num_trees; ++tree_id) {
                Tree* tree = forest[tree_id];
                vector<float> probs = tree->predict_one(data[data_index]);
                for (int i = 0; i < num_classes; ++i) {
                    avg_probs[i] += probs[i];
                }
            }
            for (int j = 0; j < num_classes; ++j) {
                avg_probs[j]/=num_trees;
                if(avg_probs[j] > max_prob){
                    max_prob = avg_probs[j];
                    label = j;
                }
            }
            y_pred[data_index]=label;
        }
        return y_pred;

    }

private:
    vector<Tree*> forest;
    int max_depth;
    float colsample;
    float rowsample;
    int num_classes;
    int num_trees;

    Bin bin;
    AttributeList attribute_list;

};
#endif //DECISIONTREE_RF_H
