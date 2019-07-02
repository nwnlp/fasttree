#include <iostream>
#include "load_data.h"
#include "rf.h"
#include <fstream>

/*void output(vector<int>&y, vector<int>&y_pred){
    string ot;
    for (int i = 0; i < y.size(); ++i) {
        ot += to_string(y[i])+" "+to_string(y_pred[i])+"\n";
    }
    std::ofstream out("../output.txt");
    out << ot;
    out.close();

}*/

int main() {

    //vector<int >res = random_pick_numbers(1000,0.01,0);
    Problem prob;
    LoadData(prob, "train.csv");
    cout<<"load data done..."<<endl;

    RandomForest clf = RandomForest(10, 15, 0.6,0.6);
    clf.fit(prob);

    //vector<vector<int>> data = bin.discrete_data(prob.X);
    //vector<int> y_pred = tree.predict(data);
    //output(prob.y, y_pred);
    return 0;
}
