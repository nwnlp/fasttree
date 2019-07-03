#include <iostream>
#include "load_data.h"
#include "rf.h"
#include <fstream>
#include <omp.h>
#include <unistd.h>

void output(vector<int>&y, vector<int>&y_pred){
    string ot;
    for (int i = 0; i < y.size(); ++i) {
        ot += to_string(y[i])+" "+to_string(y_pred[i])+"\n";
    }
    std::ofstream out("output.txt");
    out << ot;
    out.close();

}
int m = 0;


void test_omp(){
    omp_lock_t writelock;
    omp_init_lock(&writelock);
    //omp_set_num_threads(10);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 1000; ++i)
    {
        //omp_set_lock(&writelock);
        sleep(3);
        m++;
        //omp_unset_lock(&writelock);

        printf("i=%d, thread_id=%d thread_nums=%d\n", i, omp_get_thread_num(), omp_get_num_threads());

    }
    cout<<m<<endl;
    omp_destroy_lock(&writelock);
}
int main() {
    //test_omp();
    //return 0;
    //vector<int >res = random_pick_numbers(1000,0.01,0);
    Problem prob_train;
    if(!LoadData(prob_train, "train.csv")){
        return -1;
    }
    Problem prob_test;
    if(!LoadData(prob_test, "test.csv")){
        return -1;
    }
    cout<<"load data done..."<<endl;
    time_t t1 = time(0);
    RandomForest clf = RandomForest(1, 1, 0.6,0.6);
    clf.fit(prob_train);
    time_t t2= time(0);
    cout<<"fit model used:"<<t2-t1<<" seconds tree nums:"<<clf.get_tree_nums()<<endl;
    vector<int> y_pred = clf.predict(prob_test);
    time_t t3= time(0);
    cout<<"pred data used:"<<t3-t2<<"seconds"<<endl;

    output(prob_test.y, y_pred);
    return 0;
}
