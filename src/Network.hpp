#pragma once
#include <vector>

struct Neuron{
    float w;
    float b;

    float calc(float);
    Neuron();
};

class Network{
private:
    Neuron hidden;
    Neuron output;

    float sigmoid(const float&);
    float sigmoid_derivate(const float&);
    float mse(const float&, const float&);
    void backpropragation(const float&, const float&, const float&);

    float learning_rate;

public:
    Network(float);

    void train(std::vector<int>, std::vector<int>, const int&);
    float predict(const float&);
};