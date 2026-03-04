#ifndef MADALINE_H
#define MADALINE_H

#include <vector>
#include <string>
using namespace std;

const int MADALINE_ROWS = 7;
const int MADALINE_COLS = 7;
const int MADALINE_PIXELS = MADALINE_ROWS * MADALINE_COLS;
const int MADALINE_INPUT_SIZE = MADALINE_PIXELS + 1;
const int MADALINE_HIDDEN_SIZE = 43;
const int MADALINE_OUTPUT_SIZE = 36;

struct Madaline {
    vector<vector<double>> W1;
    vector<vector<double>> W2;
};

struct MadalineOutput {
    vector<double> Zin;
    vector<int> Z;
    vector<double> Yin;
    vector<int> Y;
};

MadalineOutput madaline_feed_forward(const Madaline& weights, const vector<int>& X);
bool madaline_update(Madaline& weights, const vector<int>& X, const vector<int>& T, MadalineOutput& out, double alpha);
void madaline_train(Madaline& network, const string& filename, double learning_rate, int max_epochs = 3301);
int madaline_classify(const vector<int>& X, const Madaline& network);
char madaline_classLabel(int idx);
 
#endif //MADALINE_H