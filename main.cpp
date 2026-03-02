#include <iostream>
#include <cstdlib>
using namespace std;

void showMenu() {
    cout << "\n=== Neural Networks Character Recognition ===\n";
    cout << "Choose an algorithm:\n";
    cout << "1. Hebb Learning Rule\n";
    cout << "2. Perceptron\n";
    cout << "3. Adaline (Adaptive Linear)\n";
    cout << "4. Madaline (Multiple Adaline)\n";
    cout << "5. MLP (Multi-Layer Perceptron)\n";
    cout << "0. Exit\n";
    cout << "Enter choice (0-5): ";
}

int main() {
    int choice;
    
    while (true) {
        showMenu();
        cin >> choice;
        
        switch (choice) {
            case 1:
                cout << "\nStarting Hebb Learning...\n";
                system("hebb_main");
                break;
                
            case 2:
                cout << "\nStarting Perceptron...\n";
                system("perceptron_main");
                break;
                
            case 3:
                cout << "\nStarting Adaline...\n";
                system("adaline_main");
                break;
                
            case 4:
                cout << "\nStarting Madaline...\n";
                system("madaline_main");
                break;
                
            case 5:
                cout << "\nStarting MLP...\n";
                system("mlp_main");
                break;
                
            case 0:
                cout << "Goodbye!\n";
                return 0;
                
            default:
                cout << "Invalid choice. Please try again.\n";
        }
        
        cout << "\nPress Enter to continue...";
        cin.ignore();
        cin.get();
    }
}