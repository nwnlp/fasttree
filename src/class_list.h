//
// Created by niw on 2019/6/28.
//

#ifndef DECISIONTREE_CLASS_LIST_H
#define DECISIONTREE_CLASS_LIST_H

#include <vector>
#include "tree_node.h"
using namespace std;
class ClassList{
public:
    bool build(vector<int>& train_data_inds, vector<int>& y, int num_classes){
        index2label.resize(y.size());
        for (int i = 0; i < train_data_inds.size(); ++i) {
            int index = train_data_inds[i];
            index2label[index] = y[index];
        }
        return true;
    }
    void initTreeNode(TreeNode* root, vector<int>& train_data_inds){
        index2TreeNode.resize(index2label.size(), nullptr);
        for (int i = 0; i < train_data_inds.size(); ++i) {
            int index = train_data_inds[i];
            index2TreeNode[index] = root;
        }
    }

    void setTreeNode(int index, TreeNode* node){
        index2TreeNode[index] = node;
    }

    TreeNode* get_node_by_index(int index){
        return index2TreeNode[index];
    }

    int get_label_by_index(int index){
        return index2label[index];
    }

    void buildHistogram(){
        //为含有样本的节点根据样本的类别建立直方图
        for (int index = 0; index < index2label.size(); ++index) {
            TreeNode* node = index2TreeNode[index];
            if(node)
                node->acc_total(index2label[index]);
        }
    }

    void clean_up(){
        index2label.clear();
        index2TreeNode.clear();
    }

private:
    vector<int> index2label;
    vector<TreeNode* > index2TreeNode;
};
#endif //DECISIONTREE_CLASS_LIST_H
