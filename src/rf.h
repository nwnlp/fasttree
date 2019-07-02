//
// Created by johnny on 2019/7/1.
//

#ifndef DECISIONTREE_RF_H
#define DECISIONTREE_RF_H

#include <vector>
#include "tree.h"
#include "attribute_list.h"
#include "class_list.h"
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
    }
    void fit(Problem& prob){

        bin.build(prob);
        cout<<"build bin done..."<<endl;

        attribute_list.build(bin, prob);
        cout<<"build attribute_list done..."<<endl;

        for (int tree_id = 0; tree_id <num_trees; ++tree_id) {
            Tree* tree = new Tree(tree_id,max_depth,colsample,rowsample);
            tree->fit(&attribute_list, prob.y);
            forest.push_back(tree);
        }
        attribute_list.clean_up();
    }

private:
    vector<Tree*> forest;
    int max_depth;
    float colsample;
    float rowsample;
    int num_trees;

    Bin bin;
    AttributeList attribute_list;

};
#endif //DECISIONTREE_RF_H
