#ifndef ADALINE_H
#define ADALINE_H

#include <vector>
#include <string>
using namespace std;

const int ADALINE_ROWS = 7;
const int ADALINE_COLS = 7;
const int ADALINE_PIXELS = ADALINE_ROWS * ADALINE_COLS;
const int ADALINE_INPUT_SIZE = ADALINE_PIXELS + 1;
const int ADALINE_NUM_CLASSES = 37;

vector<double> adaline_forward(const vector<int>& X, const vector<vector<double>>& W);
pair<bool, double> adaline_update(const vector<int>& X, vector<vector<double>>& W, const vector<double>& Y, const vector<int>& T);
void adaline_train(vector<vector<double>>& W, const string& filename, double user_threshold, double user_learning_rate, double user_tolerance);
int adaline_classify(const vector<int>& X, const vector<vector<double>>& W);
char adaline_classLabel(int idx);

#endif