// create_c_vector.cc
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

vector<int>* new_int_vector(int n){
    return new vector<int>(n, 0);
}
void int_vector_delete(vector<int>* v){
    delete v;
}
int int_vector_getitem(vector<int>* v, int i){
    return (*v)[i];
}
void int_vector_setitem(vector<int>* v, int i, int x){
    (*v)[i] = x;
}

vector<double>* new_double_vector(int n){
    return new vector<double>(n, 0);
}
void double_vector_delete(vector<double>* v){
    delete v;
}
double double_vector_getitem(vector<double>* v, int i){
    return (*v)[i];
}
void double_vector_setitem(vector<double>* v, int i, double x){
    (*v)[i] = x;
}

vector<vector<double>>* new_double_double_vector(int n, int m){
    return new vector<vector<double>>(n, vector<double>(m, 0));
}
void double_double_vector_delete(vector<vector<double>>* v){
    delete v;
}
double double_double_vector_getitem(vector<vector<double>>* v, int i, int j){
    return (*v)[i][j];
}
void double_double_vector_setitem(vector<vector<double>>* v, int i, int j, double x){
    (*v)[i][j] = x;
}
