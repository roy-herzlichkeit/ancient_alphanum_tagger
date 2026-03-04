#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <random>
using namespace std;

const int MLP_ROWS = 7;
const int MLP_COLS = 7;
const int MLP_PIXELS = MLP_ROWS * MLP_COLS;
const int MLP_INPUT_SIZE = MLP_PIXELS + 1;
const int MLP_HIDDEN_SIZE = 67;
const int MLP_OUTPUT_SIZE = 36;

struct MLP {
    vector<vector<double>> W, V; 
    double learning_rate = 0.1;
};

struct MLP_Outputs {
    vector<double> Yin, Z, Y;
};

inline double bipolar_sigmoid(double x) { return 2.0 / (1.0 + exp(-x)) - 1.0; }

MLP_Outputs mlp_feed_forward(const MLP& network, const vector<int>& X);
void mlp_backpropagation(const vector<double>& T, MLP& network, const MLP_Outputs& out, const vector<int>& X);
void mlp_train(MLP& network, const string& filename, double user_learning_rate, int user_max_epochs);
int mlp_classify(const vector<int>& X, const MLP& network);
char mlp_classLabel(int idx);

#endif //MLP_H