#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
using namespace std;

const int MLP_ROWS = 7;
const int MLP_COLS = 7;
const int MLP_PIXELS = MLP_ROWS * MLP_COLS;
const int MLP_INPUT_SIZE = MLP_PIXELS + 1;
const int MLP_HIDDEN_SIZE = 20;
const int MLP_OUTPUT_SIZE = 36;

struct MLP {
    vector<vector<double>> W1, W2;  
    vector<double> b1, b2;  
};

void mlp_train(MLP& network, const string& filename);
int mlp_classify(const vector<int>& X, const MLP& network);
char mlp_classLabel(int idx);

#endif