#include "adaline.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <random>

vector<double> adaline_forward(const vector<int>& X, const vector<vector<double>>& W) {
    vector<double> Y(ADALINE_NUM_CLASSES, 0.0);
    for (int j = 0; j < ADALINE_NUM_CLASSES; j++)
        for (int i = 0; i < ADALINE_INPUT_SIZE; i++)
            Y[j] += X[i] * W[j][i];
    return Y;
}

pair<bool, double> adaline_update(const vector<int>& X, vector<vector<double>>& W, const vector<double>& Y, const vector<int>& T, double lr, double tolerance) {
    bool flag = true;
    double error = 0.0;
    for (int j = 0; j < ADALINE_NUM_CLASSES; j++) {
        double err = T[j] - Y[j];
        error += err * err;
        double lre = lr * err;
        for (int i = 0; i < ADALINE_INPUT_SIZE; i++) {
            double del = lre * X[i];
            W[j][i] += del;
            flag &= (fabs(del) < tolerance);
        }
    }
    return {flag, error};
}

void adaline_train(vector<vector<double>>& W, const string& filename, double user_threshold, double user_learning_rate, double user_tolerance) {
    const double lr        = user_learning_rate / ADALINE_INPUT_SIZE;
    const double tolerance = user_tolerance;
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
        vector<int> X(ADALINE_INPUT_SIZE);
        {
            size_t pos = 0;
            for (int i = 0; i < ADALINE_INPUT_SIZE; i++) {
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
        vector<int> T(ADALINE_NUM_CLASSES);
        {
            size_t pos = 0;
            for (int i = 0; i < ADALINE_NUM_CLASSES; i++) {
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

    if (samples.empty()) {
        cerr << "Error: no samples loaded!" << endl;
        return;
    }

    int epoch = 0, max_epochs = 10007;
    double prev_E = 1e30, E = 0.0;
    mt19937 rng((unsigned)time(nullptr));

    while (epoch < max_epochs) {
        E = 0.0;
        shuffle(samples.begin(), samples.end(), rng);
        for (auto& s : samples) {
            vector<double> Y = adaline_forward(s.X, W);
            pair<bool, double> tmp = adaline_update(s.X, W, Y, s.T, lr, tolerance);
            E += tmp.second;
        }
        E /= double(samples.size());
        cout << "Epoch " << epoch + 1 << " / " << max_epochs << "  RMSE: " << sqrt(E) << "\n";
        if (fabs(prev_E - E) < tolerance) {
            epoch++;
            break;
        }
        prev_E = E;
        epoch++;
    }
    double RMSE = sqrt(E);
    cout << "Training finished at epoch " << epoch << " / " << max_epochs << endl;
    cout << "RMSE: " << RMSE << endl;
}

int adaline_classify(const vector<int>& X, const vector<vector<double>>& W) {
    vector<double> net = adaline_forward(X, W);
    int best = 0;
    for (int j = 1; j < (int)net.size(); j++)
        if (net[j] > net[best])
            best = j;
    return best;
}

char adaline_classLabel(int idx) {
    if (idx < 26) return 'A' + idx;
    if (idx < 36) return '0' + (idx - 26);
    return '?';
}