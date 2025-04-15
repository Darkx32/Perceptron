#include "Network.hpp"
#include <iostream>

int main(){
    std::vector<int> input = {0, 1};
    std::vector<int> output = {1, 0};

    Network perceptron(0.01f);
    perceptron.train(input, output, 10000);
    
    std::cout << perceptron.predict(0) << "\n";
    std::cout << perceptron.predict(1) << "\n";

    return 0;
}