#include "mlp.h"
#include <iostream>

void mlp_train(MLP& network, const string& filename) {
    cout << "MLP training - TO BE IMPLEMENTED\n";
}

int mlp_classify(const vector<int>& X, const MLP& network) {
    cout << "MLP classification - TO BE IMPLEMENTED\n";
    return 0;
}

char mlp_classLabel(int idx) {
    if (idx < 26) return 'A' + idx;
    return '0' + (idx - 26);
}