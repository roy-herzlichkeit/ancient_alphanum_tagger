#include "madaline.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>

double learning_rate;

MadalineOutput madaline_feed_forward(const Madaline& weights, const vector<int>& X) {
    MadalineOutput out;
    out.Zin.resize(MADALINE_HIDDEN_SIZE, 0.0);
    out.Z.resize(MADALINE_HIDDEN_SIZE, 0);
    out.Yin.resize(MADALINE_OUTPUT_SIZE, 0.0);
    out.Y.resize(MADALINE_OUTPUT_SIZE, 0);

    // PhASE1
    for (int i = 0; i < MADALINE_HIDDEN_SIZE; i++) {
        for (int j = 0; j < MADALINE_INPUT_SIZE; j++)
            out.Zin[i] += weights.W1[j][i] * X[j];
        out.Z[i] = (out.Zin[i] < 0) ? -1 : 1;
    }

    // PHASE2
    for (int i = 0; i < MADALINE_OUTPUT_SIZE; i++) {
        for (int j = 0; j < MADALINE_HIDDEN_SIZE; j++)
            out.Yin[i] += weights.W2[j][i] * out.Z[j];
        out.Y[i] = (out.Yin[i] < 0) ? -1 : 1;
    }

    return out;
}

bool madaline_update(Madaline& weights, const vector<int>& X, const vector<int>& T, MadalineOutput& out, double alpha) {
    bool converged = true;
    for (int i = 0; i < MADALINE_OUTPUT_SIZE; i++) {
        if (out.Y[i] != T[i]) {
            converged = false;
            if (T[i] == 1) {
                // Find hidden unit with Z = -1 and Zin ~ 0
                int minIdx = -1;
                double minAbs = 1e9;
                for (int j = 0; j < MADALINE_HIDDEN_SIZE; j++) {
                    if (out.Z[j] == -1 && fabs(out.Zin[j]) < minAbs) {
                        minAbs = fabs(out.Zin[j]);
                        minIdx = j;
                    }
                }
                if (minIdx != -1) {
                    for (int k = 0; k < MADALINE_INPUT_SIZE; k++)
                        weights.W1[k][minIdx] += alpha * (1 - out.Zin[minIdx]) * X[k];
                }
            } else {
                // Update all hidden units with Z = 1
                for (int j = 0; j < MADALINE_HIDDEN_SIZE; j++) {
                    if (out.Z[j] == 1) {
                        for (int k = 0; k < MADALINE_INPUT_SIZE; k++)
                            weights.W1[k][j] += alpha * (-1 - out.Zin[j]) * X[k];
                    }
                }
            }
        }
    }
    return converged;
}

void madaline_train(Madaline& network, const string& filename, double user_learning_rate, int user_max_epochs) {
    learning_rate = user_learning_rate;
    int max_epochs = (user_max_epochs <= 0) ? 3301 : user_max_epochs;

    // Initialize weight matrices with small random values
    srand((unsigned)time(nullptr));
    auto randW = []() { return ((double)rand() / RAND_MAX) * 0.1 - 0.05; };
    network.W1.assign(MADALINE_INPUT_SIZE,  vector<double>(MADALINE_HIDDEN_SIZE, 0.0));
    network.W2.assign(MADALINE_HIDDEN_SIZE, vector<double>(MADALINE_OUTPUT_SIZE, 0.0));
    for (int i = 0; i < MADALINE_INPUT_SIZE;  i++)
        for (int j = 0; j < MADALINE_HIDDEN_SIZE; j++)
            network.W1[i][j] = randW();
    for (int i = 0; i < MADALINE_HIDDEN_SIZE; i++)
        for (int j = 0; j < MADALINE_OUTPUT_SIZE; j++)
            network.W2[i][j] = randW();

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
        vector<int> X(MADALINE_INPUT_SIZE);
        {
            size_t pos = 0;
            for (int i = 0; i < MADALINE_INPUT_SIZE; i++) {
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
        vector<int> T(MADALINE_OUTPUT_SIZE);
        {
            size_t pos = 0;
            for (int i = 0; i < MADALINE_OUTPUT_SIZE; i++) {
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

    int epoch = 0, sn = samples.size();
    bool converged = false;

    while (epoch < max_epochs && !converged) {
        converged = true;
        for (auto& s : samples) {
            MadalineOutput out = madaline_feed_forward(network, s.X);
            if (!madaline_update(network, s.X, s.T, out, learning_rate))
                converged = false;
        }
        cout << "Epoch " << epoch + 1 << " / " << max_epochs << endl;
        epoch++;
    }
    cout << "Training finished at epoch " << epoch << " / " << max_epochs << endl;
    if (converged)
        cout << "Network converged!" << endl;
}

int madaline_classify(const vector<int>& X, const Madaline& network) {
    MadalineOutput out = madaline_feed_forward(network, X);
    for (int i = 0; i < MADALINE_OUTPUT_SIZE; i++)
        if (out.Y[i] == 1)
            return i;
    int best = 0;
    for (int j = 1; j < MADALINE_OUTPUT_SIZE; j++)
        if (out.Yin[j] > out.Yin[best])
            best = j;
    return best;
}

char madaline_classLabel(int idx) {
    if (idx < 26) return 'A' + idx;
    return '0' + (idx - 26);
}