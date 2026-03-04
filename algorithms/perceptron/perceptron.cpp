#include "perceptron.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <ctime>

vector<double> perceptron_forward(const vector<int>& X, const vector<vector<double>>& W) {
    int m = (int)W.size(), n = (int)W[0].size();
    vector<double> Y(m, 0.0);
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            Y[j] += X[i] * W[j][i];
    return Y;
}

vector<int> perceptron_activate(const vector<double>& net, double threshold) {
    int n = net.size();
    vector<int> Y(n, 0);
    for (int i = 0; i < n; i++)
        Y[i] = (net[i] >= threshold) ? 1 : -1;
    return Y;
}

bool perceptron_comparison(const vector<int>& Y, const vector<int>& T) {
    for (int i = 0, n = Y.size(); i < n; i++)
        if (Y[i] != T[i])
            return false;
    return true;
}

void perceptron_update(const vector<int>& X, vector<vector<double>>& W, const vector<int>& activated, const vector<int>& T, double lr) {
    int m = (int)T.size(), n = (int)X.size();
    for (int j = 0; j < m; j++) {
        double d = lr * (T[j] - activated[j]);
        for (int i = 0; i < n; i++)
            W[j][i] += d * X[i];
    }
}

void perceptron_train(vector<vector<double>>& W, const string& filename, double user_threshold, double user_learning_rate, int max_epochs) {
    const double threshold    = user_threshold;
    const double learning_rate = user_learning_rate;
    struct Sample { vector<int> X; vector<int> T; };
    vector<Sample> samples;

    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: cannot open " << filename << endl;
        return;
    }

    string line;
    while (getline(fin, line)) {
        if (line.empty())
            continue;
        vector<int> X(PERCEPTRON_INPUT_SIZE);
        {
            size_t pos = 0;
            for (int i = 0; i < PERCEPTRON_INPUT_SIZE; i++) {
                while (pos < line.size() && line[pos] == ' ')
                    pos++;
                int sign = 1;
                if (pos < line.size() && line[pos] == '-') { sign = -1; ++pos; }
                int num = 0;
                while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9')
                    num = num * 10 + (line[pos++] - '0');
                X[i] = sign * num;
            }
        }
        if (!getline(fin, line))
            break;
        vector<int> T(PERCEPTRON_NUM_CLASSES);
        {
            size_t pos = 0;
            for (int i = 0; i < PERCEPTRON_NUM_CLASSES; i++) {
                while (pos < line.size() && line[pos] == ' ') ++pos;
                int sign = 1;
                if (pos < line.size() && line[pos] == '-') { sign = -1; ++pos; }
                int num = 0;
                while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9')
                    num = num * 10 + (line[pos++] - '0');
                T[i] = sign * num;
            }
        }
        samples.push_back({X, T});
    }
    fin.close();

    mt19937 rng((unsigned)time(nullptr));
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        bool converged = true;
        shuffle(samples.begin(), samples.end(), rng);
        for (auto& s : samples) {
            vector<double> net = perceptron_forward(s.X, W);
            vector<int> Y = perceptron_activate(net, threshold);
            if (!perceptron_comparison(Y, s.T)) {
                converged = false;
                perceptron_update(s.X, W, Y, s.T, learning_rate);
            }
        }
        if (converged) {
            cout << "Converged at epoch " << epoch + 1 << endl;
            return;
        }
    }
    cout << "Reached max epochs (" << max_epochs << ")" << endl;
}

int perceptron_classify(const vector<int>& X, const vector<vector<double>>& W) {
    vector<double> net = perceptron_forward(X, W);
    int best = 0;
    for (int j = 1; j < (int)net.size(); j++)
        if (net[j] > net[best])
            best = j;
    return best;
}

char perceptron_classLabel(int idx) {
    if (idx < 26) return 'A' + idx;
    if (idx < 36) return '0' + (idx - 26);
    return '?';
}