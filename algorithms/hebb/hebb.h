#ifndef HEBB_H
#define HEBB_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
using namespace std;

const int HEBB_ROWS   = 7;
const int HEBB_COLS   = 7;
const int HEBB_PIXELS = HEBB_ROWS * HEBB_COLS;
const int INPUT_SIZE  = HEBB_PIXELS + 1;
const int NUM_CLASSES = 37;


vector<int> multiply(const vector<int>& X, const vector<vector<int>>& W);

vector<int> activate(const vector<int>& net);

void update(const vector<int>& X, vector<vector<int>>& W, const vector<int>& Y);

void train(vector<vector<int>>& W, const string& filename);

int classify(const vector<int>& X, const vector<vector<int>>& W);

char classLabel(int idx);

#endif //HEBB_H