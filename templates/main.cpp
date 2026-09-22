#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>

// ── Daily-use C++ template ───────────────────────────────────────────────────
// Build:  g++ -std=c++20 -Wall -Wextra main.cpp -o app
// Run:    ./app [args]          (or: ./app < input.txt)

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::cout << "arg " << i << ": " << argv[i] << '\n';
    }

    std::string name = "world";
    std::cout << "hello " << name << '\n';

    // read from stdin (redirect a file: ./app < input.txt)
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::cout << "> " << line << '\n';
    }

    return 0;
}
