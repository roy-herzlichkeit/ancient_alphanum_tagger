#include "mlp.h"

MLP_Outputs mlp_feed_forward(const MLP& network, const vector<int>& X) {
    MLP_Outputs out;
    out.Z.resize(MLP_HIDDEN_SIZE, 0.0);
    out.Yin.resize(MLP_OUTPUT_SIZE, 0.0);
    out.Y.resize(MLP_OUTPUT_SIZE, 0.0);

    // Phase 1
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        double zin = 0.0;
        for (int j = 0; j < MLP_INPUT_SIZE; j++)
            zin += network.W[i][j] * X[j];
        out.Z[i] = bipolar_sigmoid(zin);
    }

    // Phase 2
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
        for (int j = 0; j < MLP_HIDDEN_SIZE; j++)
            out.Yin[i] += network.V[j][i] * out.Z[j];
        out.Y[i] = bipolar_sigmoid(out.Yin[i]);
    }

    return out;
}

void mlp_backpropagation(const vector<double>& T, MLP& network, const MLP_Outputs& out, const vector<int>& X) {
    const double lr = network.learning_rate;

    vector<double> deltaV(MLP_OUTPUT_SIZE);
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++)
        deltaV[i] = (T[i] - out.Y[i]) * 0.5 * (1.0 - out.Y[i] * out.Y[i]);

    vector<double> deltaW(MLP_HIDDEN_SIZE, 0.0);
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        for (int j = 0; j < MLP_OUTPUT_SIZE; j++) {
            deltaW[i] += deltaV[j] * network.V[i][j];
            network.V[i][j] += lr * deltaV[j] * out.Z[i];
        }
        deltaW[i] *= 0.5 * (1.0 - out.Z[i] * out.Z[i]);
    }

    for (int i = 0; i < MLP_HIDDEN_SIZE; i++)
        for (int j = 0; j < MLP_INPUT_SIZE; j++)
            network.W[i][j] += lr * deltaW[i] * X[j];
}

void mlp_train(MLP& network, const string& filename, double user_learning_rate, int user_max_epochs) {
    network.learning_rate = user_learning_rate;

    srand((unsigned)time(nullptr));
    double limW = sqrt(6.0 / (MLP_INPUT_SIZE + MLP_HIDDEN_SIZE));
    double limV = sqrt(6.0 / (MLP_HIDDEN_SIZE + MLP_OUTPUT_SIZE));
    auto randRange = [](double lim) { return ((double)rand() / RAND_MAX) * 2.0 * lim - lim; };
    network.W.assign(MLP_HIDDEN_SIZE, vector<double>(MLP_INPUT_SIZE, 0.0));
    network.V.assign(MLP_HIDDEN_SIZE, vector<double>(MLP_OUTPUT_SIZE, 0.0));
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        for (int j = 0; j < MLP_INPUT_SIZE; j++)
            network.W[i][j] = randRange(limW);
        for (int j = 0; j < MLP_OUTPUT_SIZE; j++)
            network.V[i][j] = randRange(limV);
    }

    struct Sample { vector<int> X; vector<double> T; };
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
        vector<int> X(MLP_INPUT_SIZE);
        {
            size_t pos = 0;
            for (int i = 0; i < MLP_INPUT_SIZE; i++) {
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
        vector<double> T(MLP_OUTPUT_SIZE);
        {
            size_t pos = 0;
            for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
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

    mt19937 rng((unsigned)time(nullptr));
    const double early_stop_thresh = 1e-10;
    double prev_mse = 1e18;

    for (int epoch = 0; epoch < user_max_epochs; epoch++) {
        shuffle(samples.begin(), samples.end(), rng);

        double mse = 0.0;
        for (auto& s : samples) {
            MLP_Outputs out = mlp_feed_forward(network, s.X);
            mlp_backpropagation(s.T, network, out, s.X);
            for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
                double e = s.T[i] - out.Y[i];
                mse += e * e;
            }
        }
        mse /= (double)(samples.size() * MLP_OUTPUT_SIZE);
        cout << "Epoch " << epoch + 1 << " / " << user_max_epochs << "  MSE: " << mse << "\n";

        if (fabs(prev_mse - mse) < early_stop_thresh) {
            cout << "Early stopping at epoch " << epoch + 1 << "\n";
            break;
        }
        prev_mse = mse;
    }
    cout << "Training complete.\n";
}

int mlp_classify(const vector<int>& X, const MLP& network) {
    MLP_Outputs out = mlp_feed_forward(network, X);
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++)
        if (out.Y[i] > 0.9)
            return i;
    int best = 0;
    for (int j = 1; j < MLP_OUTPUT_SIZE; j++)
        if (out.Yin[j] > out.Yin[best])
            best = j;
    return best;
}

char mlp_classLabel(int idx) {
    if (idx < 26) return 'A' + idx;
    if (idx < 36) return '0' + (idx - 26);
    return '?';
}