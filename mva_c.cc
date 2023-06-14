#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <tuple>
#include<stdio.h>
#include<stdlib.h>

using namespace std;

class Mva {
    public:
        bool complete_probs;
        int probs_size;

        vector<int>* N = new vector<int>();
        vector<int>* refs = new vector<int>();
        vector<vector<double>>* visits = new vector<vector<double>>();
        vector<int>* caps = new vector<int>();
        vector<double>* service_times = new vector<double>();

        vector<double>* result_nrs = new vector<double>();
        vector<vector<double>>* result_waits = new vector<vector<double>>();
        vector<double>* result_throughputs = new vector<double>();
        vector<double>* result_utils = new vector<double>();
        vector<vector<double>>* result_probs = new vector<vector<double>>();

        Mva(int R, int M, int probs, bool complete) {
            N->resize(R, 0);
            refs->resize(R, 0);
            visits->resize(M, vector<double>(R, 0));
            caps->resize(M, 0);
            service_times->resize(M, 0);

            complete_probs = complete;
            probs_size = probs;

            result_nrs->resize(M, 0);
            result_waits->resize(M, vector<double>(R, 0));
            result_throughputs->resize(R, 0);
            result_utils->resize(M, 0);
            result_probs->resize(M, vector<double>(probs_size, 0));
        }
        ~Mva() {
            delete N;
            delete refs;
            delete visits;
            delete caps;
            delete service_times;

            delete result_nrs;
            delete result_waits;
            delete result_throughputs;
            delete result_utils;
            delete result_probs;
        }
    int total_customers(vector<int>* N, int N_) {
        int totalN_ = 0;
        int cur = N_;
        for (int r = 0; r < (*N).size(); r++) {
            int Nr = (*N)[r];
            totalN_ += cur % (Nr + 1);
            cur /= (Nr + 1);
        }
        return totalN_;
    }

    void mva_c() {
        int R = (*N).size();
        int M = (*visits).size();

        vector<double> mus;
        for (int i = 0; i < (*service_times).size(); i++) {
            mus.push_back(1 / (*service_times)[i]);
        }

        for (int r = 0; r < R; r++) {
            double ref = (*visits)[(*refs)[r]][r];
            for (int i = 0; i < M; i++) {
                (*visits)[i][r] /= ref;
            }
        }

        vector<vector<int>> Rs;
        for (int i = 0; i < M; i++) {
            vector<int> temp;
            for (int r = 0; r < R; r++) {
                if ((*visits)[i][r] > 0) {
                    temp.push_back(r);
                }
            }
            Rs.push_back(temp);
        }

        vector<vector<int>> Ms;
        for (int r = 0; r < R; r++) {
            vector<int> temp;
            for (int i = 0; i < M; i++) {
                if ((*visits)[i][r] > 0) {
                    temp.push_back(i);
                }
            }
            Ms.push_back(temp);
        }

        int nr_of_states = 1;
        for (int i = 0; i < (*N).size(); i++) {
            nr_of_states *= ((*N)[i] + 1);
        }

        int totalN = 0;
        for (int i = 0; i < (*N).size(); i ++) {
            totalN += (*N)[i];
        }

        int max_caps = *max_element((*caps).begin(), (*caps).end());
        vector<vector<double>> nrs(M, vector<double>(nr_of_states, 0));
        vector<vector<double>> utils(M, vector<double>(nr_of_states, 0));
        vector<vector<vector<double>>> probs(M, vector<vector<double>>(probs_size, vector<double>(nr_of_states, 0)));
        vector<vector<vector<double>>> waits(M, vector<vector<double>>(R, vector<double>(nr_of_states, 0)));
        vector<vector<double>> throughputs(R, vector<double>(nr_of_states, 0));

        for (int i = 0; i < M; i++) {
            probs[i][0][0] = 1;
        }

        vector<int> N_products(R, 0);
        int cur = 1;
        for (int r = 0; r < R; r++) {
            N_products[r] = cur;
            cur *= ((*N)[r] + 1);
        }

        for (int N_ = 0; N_ < nr_of_states; N_++) {
            int totalN_ = total_customers(N, N_);
            int cur = N_;

            for (int r = 0; r < (*N).size(); r++) {
                int Nr = (*N)[r];
                int Nr_ = cur % (Nr + 1);
                cur /= (Nr + 1);

                if (Nr_ != 0) {
                    int r_cus_removal = N_ - N_products[r];

                    for (int j = 0; j < Ms[r].size(); j++) {
                        int i = Ms[r][j];
                        int ci = (*caps)[i];

                        double sum = 0;
                        for (int n = 0; n < 1 + min(ci - 2, totalN_ - 1); n++) {
                            sum += (ci - n - 1) * probs[i][n][r_cus_removal];
                        }
                        waits[i][r][N_] = 1 / (ci * mus[i]) * (1 + nrs[i][r_cus_removal] + sum);
                    }
                    double sum = 0;
                    for (int j = 0; j < Ms[r].size(); j++) {
                        int i = Ms[r][j];
                        sum += (*visits)[i][r] * waits[i][r][N_];
                    }
                    throughputs[r][N_] = Nr_ / sum;
                }
            }

            for (int i = 0; i < M; i++) {
                double sum_nrs = 0;
                double sum_utils = 0;
                for (int s = 0; s < Rs[i].size(); s++) {
                    int r = Rs[i][s];
                    sum_nrs += (*visits)[i][r] * throughputs[r][N_] * waits[i][r][N_];
                    sum_utils += throughputs[r][N_] * (*visits)[i][r];
                }
                nrs[i][N_] = sum_nrs;
                utils[i][N_] = 1 / mus[i] * sum_utils;

                int ci = (*caps)[i];
                // int probs_size = totalN_ + 1;
                // if (complete_probs == false) {
                //     probs = min(ci, probs);
                // }
                for (int n = 1; n < probs_size; n++) {
                    double sum = 0;
                    for (int s = 0; s < Rs[i].size(); s++) {
                        int r = Rs[i][s];
                        if (N_ - N_products[r] >= 0) {
                            sum += (*visits)[i][r] * throughputs[r][N_] * probs[i][n - 1][N_ - N_products[r]];
                        }
                    }
                    probs[i][n][N_] = 1.0 / (min(n, ci) * mus[i]) * sum;
                }
                double sum = 0;
                for (int n = 1; n < 1 + min(ci - 1, totalN_); n++) {
                    sum += (ci - n) * probs[i][n][N_];
                }
                // cout << "sum " << sum << "\n";
                probs[i][0][N_] = 1 - 1 / ((double) ci) * (utils[i][N_] + sum);
            }
        }

        int final_state = nr_of_states - 1;
        for (int i = 0; i < M; i++) {
            (*result_nrs)[i] = nrs[i][final_state];
            (*result_utils)[i] = utils[i][final_state];
            for (int n = 0; n < (*result_probs)[i].size(); n++) {
                (*result_probs)[i][n] = probs[i][n][final_state];
            }
            for (int r = 0; r < R; r++){
                (*result_waits)[i][r] = waits[i][r][final_state];
            }
        }
        for (int r = 0; r < R; r++) {
            (*result_throughputs)[r] = throughputs[r][final_state];
        }
    }
};


extern "C" {
    Mva* Mva_new(int R, int M, int probs_size, int complete_probs){ return new Mva(R, M, probs_size, complete_probs); }
    // void mva_c(Mva* mva,
    //            vector<int>* N, vector<int>* refs,
    //            vector<vector<double>>* visits,
    //            vector<int>* caps, vector<double>* service_times
    //            ){ mva->mva_c(N, refs, visits, caps, service_times); }
    void N_set(Mva* mva, int i, int x) {(*mva->N)[i] = x;}
    void refs_set(Mva* mva, int i, int x) {(*mva->refs)[i] = x;}
    void visits_set(Mva* mva, int i, int j, double x) {(*mva->visits)[i][j] = x;}
    void caps_set(Mva* mva, int i, int x) {(*mva->caps)[i] = x;}
    void service_times_set(Mva* mva, int i, double x) {(*mva->service_times)[i] = x;}
    double nrs_get(Mva* mva, int i) {return (*mva->result_nrs)[i];}
    double waits_get(Mva* mva, int i, int j) {return (*mva->result_waits)[i][j];}
    double throughputs_get(Mva* mva, int i) {return (*mva->result_throughputs)[i];}
    double utils_get(Mva* mva, int i) {return (*mva->result_utils)[i];}
    double probs_get(Mva* mva, int i, int j) {return (*mva->result_probs)[i][j];}

    void compute(Mva* mva) {mva->mva_c();}
};
