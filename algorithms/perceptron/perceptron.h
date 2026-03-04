#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <string>
using namespace std;

const int PERCEPTRON_ROWS   = 7;
const int PERCEPTRON_COLS   = 7;
const int PERCEPTRON_PIXELS = PERCEPTRON_ROWS * PERCEPTRON_COLS;
const int PERCEPTRON_INPUT_SIZE  = PERCEPTRON_PIXELS + 1;
const int PERCEPTRON_NUM_CLASSES = 37;

vector<double> perceptron_forward(const vector<int>& X, const vector<vector<double>>& W);
void perceptron_train(vector<vector<double>>& W, const string& filename, double user_threshold, double user_learning_rate, int max_epochs = 3301);
int perceptron_classify(const vector<int>& X, const vector<vector<double>>& W);
char perceptron_classLabel(int idx);

#endif // PERCEPTRON_H