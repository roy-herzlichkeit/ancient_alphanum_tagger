#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

#ifdef _WIN32
#  include <windows.h>
static string exeDir() {
    char buf[MAX_PATH];
    GetModuleFileNameA(NULL, buf, MAX_PATH);
    string path(buf);
    size_t pos = path.find_last_of("\\/");
    return (pos != string::npos) ? path.substr(0, pos + 1) : ".\\";
}
static void cdExeDir(const string& dir) { SetCurrentDirectoryA(dir.c_str()); }
static int launch(const string& dir, const char* name) {
    string cmd = "\"" + dir + name + ".exe\"";
    return system(cmd.c_str());
}
#else
#  include <unistd.h>
#  include <limits.h>
static string exeDir() {
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len == -1) return "./";
    buf[len] = '\0';
    string path(buf);
    size_t pos = path.find_last_of('/');
    return (pos != string::npos) ? path.substr(0, pos + 1) : "./";
}
static void cdExeDir(const string& dir) { chdir(dir.c_str()); }
static int launch(const string& dir, const char* name) {
    string cmd = "\"" + dir + name + "\"";
    return system(cmd.c_str());
}
#endif

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
    // Change working directory to the folder containing neural_selector.exe
    // so that all sub-executables find data/training.txt correctly.
    string dir = exeDir();
    cdExeDir(dir);

    int choice;
    while (true) {
        showMenu();
        if (!(cin >> choice)) {
            cin.clear();
            cin.ignore(1024, '\n');
            continue;
        }

        switch (choice) {
            case 1:
                cout << "\nStarting Hebb Learning...\n";
                launch(dir, "hebb_main");
                break;
            case 2:
                cout << "\nStarting Perceptron...\n";
                launch(dir, "perceptron_main");
                break;
            case 3:
                cout << "\nStarting Adaline...\n";
                launch(dir, "adaline_main");
                break;
            case 4:
                cout << "\nStarting Madaline...\n";
                launch(dir, "madaline_main");
                break;
            case 5:
                cout << "\nStarting MLP...\n";
                launch(dir, "mlp_main");
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
