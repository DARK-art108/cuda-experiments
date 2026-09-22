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

    // std::string name = "world";
    // std::cout << "hello " << name << '\n';
    int arr[5] = {1,2,3,4,5};
    int arr_[5] = {2,3,4,5,6};
    int arr_k[5];


    for(int i=0;i<5;i++)
    {
        arr_k[i] = arr[i] + arr_[i];
    }

    for(int i=0;i<5;i++)
    {
        std::cout<<arr_k[i];

    }

    // read from stdin (redirect a file: ./app < input.txt)
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::cout << "> " << line << '\n';
    }

    return 0;
}
