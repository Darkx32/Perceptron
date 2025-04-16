#pragma once
#include <vector>

struct Layer
{
    std::vector<float> weights;
    float b;
    int n_neurons;
    int n_past_neurons;

    Layer() = default;
    Layer(int n_neurons, int n_past_neurons = 1);
    std::vector<float> foward(std::vector<float> x);
    std::vector<float> back(float delta, std::vector<float> past_weights);
};


class Network{
private:
    Layer hidden;
    Layer out;

    float sigmoid(const float&);
    float sigmoid_derivate(const float&);
    float mse(const float&, const float&);
    void backpropragation(std::vector<float> inputs, const float& pred, const float& real_pred);

    float learning_rate;

public:
    Network(float);

    void train(std::vector<std::pair<float,float>> input_data, std::vector<float> predict_data, const int &epochs);
    float predict(const float&, const float&);
};