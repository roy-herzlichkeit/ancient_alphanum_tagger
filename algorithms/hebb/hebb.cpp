#include "hebb.h"
  
// Y = X * W  
vector<int> multiply(const vector<int>& X, const vector<vector<int>>& W) {
    int m = (int)W.size();
    int n = (int)X.size();
    vector<int> Y(m, 0);
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            Y[j] += X[i] * W[j][i];
    return Y;
}
 
// Y = S
vector<int> activate(const vector<int>& net) {
    vector<int> Y(net.size());
    for (size_t j = 0; j < net.size(); j++)
        Y[j] = (net[j] > 0) ? 1 : -1;
    return Y;
}

// Wi(new) = Wi(old) + XiY  
void update(const vector<int>& X, vector<vector<int>>& W, const vector<int>& Y) {
    int n = (int)X.size();
    int m = (int)Y.size();
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            W[j][i] += X[i] * Y[j];
}
 
void train(vector<vector<int>>& W, const string& filename) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: cannot open " << filename << endl;
        return;
    }

    string line;
    while (getline(fin, line)) { 
        if (line.empty()) 
            continue; 
        vector<int> X(INPUT_SIZE);
        {
            size_t pos = 0;
            for (int i = 0; i < INPUT_SIZE; i++) {
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
        vector<int> Y(NUM_CLASSES);
        {
            size_t pos = 0;
            for (int i = 0; i < NUM_CLASSES; i++) {
                while (pos < line.size() && line[pos] == ' ') ++pos;
                int sign = 1;
                if (pos < line.size() && line[pos] == '-') { sign = -1; ++pos; }
                int num = 0;
                while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9')
                    num = num * 10 + (line[pos++] - '0');
                Y[i] = sign * num;
            }
        } 
        update(X, W, Y);
    }
    fin.close();
    cout << "Training complete." << endl;
}
 
int classify(const vector<int>& X, const vector<vector<int>>& W) {
    vector<int> net = multiply(X, W);  
    int best = 0;
    for (int j = 1; j < (int)net.size(); j++) {
        if (net[j] > net[best])
            best = j;
    }
    return best;
}
 
char classLabel(int idx) {
    if (idx < 26) 
        return 'A' + idx; 
    if (idx < 36)
        return '0' + (idx - 26); 
    return '?';
}