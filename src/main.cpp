#include "Network.hpp"
#include <iostream>

int main(){
    std::vector<std::pair<float,float>> input = {{0.f, 1.f}, {0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}};
    std::vector<float> output = {0.f, 0.f, 0.f, 1.f};

    Network perceptron(0.01f);
    perceptron.train(input, output, 5000);

    std::cout << perceptron.predict(0.f, 1.f) << std::endl;
    

    return 0;
}