#include "madaline.h"
#include <iostream>

void madaline_train(Madaline& network, const string& filename) {
    cout << "MADALINE training - TO BE IMPLEMENTED\n";
}

int madaline_classify(const vector<int>& X, const Madaline& network) {
    cout << "MADALINE classification - TO BE IMPLEMENTED\n";
    return 0;
}

char madaline_classLabel(int idx) {
    if (idx < 26) return 'A' + idx;
    return '0' + (idx - 26);
}