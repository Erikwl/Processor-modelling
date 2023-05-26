#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <tuple>
#include<stdio.h>
#include<stdlib.h>

using namespace std;

class Mva{
    public:
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

        tuple<vector<double>,
            vector<vector<double>>,
            vector<double>,
            vector<double>,
            vector<vector<double>>> mva_c(vector<int>* N, vector<int>* refs,
                                          vector<vector<double>>* visits,
                                          vector<int>* caps, vector<double>* service_times) {
            // vector<vector<double>>> mva_c(vector<int> N, vector<int> refs,
            //                             vector<vector<double>> visits,
            //                             vector<int> caps, vector<double> service_times) {
            vector<double> mus;
            for (int i = 0; i < (*service_times).size(); i++) {
                mus.push_back(1 / (*service_times)[i]);
            }
            int R = (*N).size();
            int M = (*refs).size();

            // Normalize visits.
            for (int r = 0; r < (*refs).size(); r++) {
                for (int i = 0; i < M; i++) {
                    (*visits)[i][r] /= (*visits)[(*refs)[r]][r];
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
            vector<vector<vector<double>>> probs(M, vector<vector<double>>(min(totalN + 1, max_caps), vector<double>(nr_of_states, 0)));
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

                    int r_cus_removal = N_ - N_products[r];

                    for (int j = 0; j < Ms[r].size(); j++) {
                        int i = Ms[r][j];
                        int ci = (*caps)[i];

                        if (Nr_ != 0) {
                            double sum = 0;
                            for (int n = 0; n < 1 + min(ci - 2, totalN_ - 1); n++) {
                                sum += (ci - n - 1) * probs[i][n][r_cus_removal];
                            }
                            waits[i][r][N_] = 1 / (ci * mus[i]) * (1 + nrs[i][r_cus_removal] + sum);
                        }
                    }
                    if (Nr != 0) {
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
                    for (int n = 1; n < 1 + min(ci - 1, totalN_); n++) {
                        double sum = 0;
                        for (int s = 0; s < Rs[i].size(); s++) {
                            int r = Rs[i][s];
                            sum += (*visits)[i][r] * throughputs[r][N_] * probs[i][n - 1][N_ - N_products[r]];
                        }
                        probs[i][n][N_] = 1.0 / ((float) min(n, ci)) * mus[i] * sum;
                    }
                    double sum = 0;
                    for (int n = 1; n < 1 + min(ci - 1, totalN_); n++) {
                        sum += (ci - n) * probs[i][n][N_];
                    }
                    probs[i][0][N_] = 1 - 1 / ((float) ci) * utils[i][N_] + sum;
                }

            }

            int final_state = nr_of_states - 1;
            vector<double> result_nrs(M, 0);
            vector<vector<double>> result_waits(M, vector<double>(R, 0));
            vector<double> result_throughputs(R, 0);
            vector<double> result_utils(M, 0);
            vector<vector<double>> result_probs(M, vector<double>(min(totalN + 1, max_caps), 0));
            for (int i = 0; i < M; i++) {
                result_nrs[i] = nrs[i][final_state];
                result_utils[i] = utils[i][final_state];
                for (int n = 0; n < result_probs[i].size(); n++) {
                    result_probs[i][n] = probs[i][n][final_state];
                }
            }
            for (int r = 0; r < R; r++) {
                result_throughputs[r] = throughputs[r][final_state];
            }

            return {result_nrs, result_waits, result_throughputs, result_utils, result_probs};
        }
};

extern "C" {
    Mva* Mva_new(){ return new Mva(); }
    void mva_c(Mva* mva,
               vector<int>* N, vector<int>* refs,
               vector<vector<double>>* visits,
               vector<int>* caps, vector<double>* service_times
               ){ mva->mva_c(N, refs, visits, caps, service_times); }
}


// int main() {
//     int n = 2;
//     vector<int> N(n, n);
//     vector<int> refs(2, 1);
//     vector<vector<double>> visits(2, vector<double>(2, 1));

//     vector<int> caps(2, 1);
//     vector<double> service_times(2, 1);
//     mva_c(N, refs, visits, caps, service_times);
//     return 0;
// }
