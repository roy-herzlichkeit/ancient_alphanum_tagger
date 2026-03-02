#ifndef MADALINE_H
#define MADALINE_H

#include <vector>
#include <string>
using namespace std;

const int MADALINE_ROWS = 7;
const int MADALINE_COLS = 7;
const int MADALINE_PIXELS = MADALINE_ROWS * MADALINE_COLS;
const int MADALINE_INPUT_SIZE = MADALINE_PIXELS + 1;
const int MADALINE_HIDDEN_SIZE = 15;
const int MADALINE_OUTPUT_SIZE = 36;

struct Madaline {
    vector<vector<double>> W1;
    vector<vector<double>> W2;
};

void madaline_train(Madaline& network, const string& filename);
int madaline_classify(const vector<int>& X, const Madaline& network);
char madaline_classLabel(int idx);
 
#endif //MADALINE_H