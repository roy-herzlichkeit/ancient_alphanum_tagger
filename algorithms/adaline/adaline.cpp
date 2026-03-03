#include "adaline.h"
#include <iostream>
#include <fstream>
#include <cmath>

double threshold, learning_rate, tolerance;
int n = ADALINE_INPUT_SIZE, m = ADALINE_NUM_CLASSES;

vector<double> adaline_forward(const vector<int>& X, const vector<vector<double>>& W) {
    vector<double> Y(m, 0.0);
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < m; j++)
            Y[j] += X[i] * W[i][j];
    return Y;
}

pair<bool, double> adaline_update(const vector<int>& X, vector<vector<double>>& W, const vector<double>& Y, const vector<int>& T) {
    bool flag = true;
    double error = 0.0;
    
    for (int j = 0; j < m; j++) {
        double err = T[j] - Y[j];
        error += err * err;
    }
    
    double lr = learning_rate / n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double del = lr * (T[j] - Y[j]) * X[i];
            W[i][j] += del;
            flag &= (fabs(del) < tolerance);
        }
    }
    return {flag, error};
}

void adaline_train(vector<vector<double>>& W, const string& filename, double user_threshold, double user_learning_rate, double user_tolerance) {
    threshold = user_threshold, learning_rate = user_learning_rate, tolerance = user_tolerance;
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
        vector<int> X(n);
        {
            size_t pos = 0;
            for (int i = 0; i < n; i++) {
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
        vector<int> T(m);
        {
            size_t pos = 0;
            for (int i = 0; i < m; i++) {
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

    int epoch = 0, max_epochs = 10007
    , sn = samples.size();
    double prev_E = 1e30, E = 0.0;

    while (epoch < max_epochs) {
        E = 0.0;
        for (auto& s : samples) {
            vector<double> Y = adaline_forward(s.X, W);
            pair<bool, double> tmp = adaline_update(s.X, W, Y, s.T);
            E += tmp.second;
        }
        E /= double(sn);
        cout << "Epoch " << epoch + 1 << " / " << max_epochs << "  RMSE: " << sqrt(E) << endl;
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
    if (idx < 37) return '0' + (idx - 26);
    return '?';
}