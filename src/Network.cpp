#include "Network.hpp"
#include <iostream>
#include <ctime>
#include <iostream>

float Neuron::calc(float x)
{
    return this->w * x + b;
}

Neuron::Neuron()
{
    this->w = rand() / RAND_MAX * 2.0f - 1.0f;
    this->b = rand() / RAND_MAX * 2.0f - 1.0f;
}

float Network::sigmoid(const float& x)
{
    return 1 / (1 + exp(-x));
}

float Network::sigmoid_derivate(const float& x)
{
    float s = this->sigmoid(x);
    return s * (1 - s);
}

float Network::mse(const float& x, const float& y)
{
    return powf(x - y, 2);
}

void Network::train(std::vector<int> input_data, std::vector<int> predict_data, const int &epochs)
{
    for(int e = 0;e < epochs;e++){
        float actual_loss = 0.0f;
        for (int i = 0;i < input_data.size();i++)
        {
            float pred = this->predict(static_cast<float>(input_data[i]));
            float loss = this->mse(static_cast<float>(predict_data[i]), pred);
            actual_loss += loss;

            this->backpropragation(static_cast<float>(input_data[i]), static_cast<float>(predict_data[i]), pred);
        }
        std::cout << "Epoch: " << e << " loss: " << actual_loss << "\n";
    }
}

float Network::predict(const float &x)
{
    float pred = hidden.calc(x);
    pred = output.calc(pred);
    return this->sigmoid(pred);
}

void Network::backpropragation(const float& x, const float& y, const float& pred)
{
    float Zh = this->hidden.calc(x);
    float Zo = this->output.calc(Zh);

    float Eo = (pred - y) * sigmoid_derivate(Zo);
    float Eh = Eo * this->output.w * Zh;

    float dWo = Eo * Zh;
    float dWh = Eh * x;

    this->hidden.w -= this->learning_rate * dWh;
    this->output.w -= this->learning_rate * dWo;
}

Network::Network(float learning_rate)
{
    this->learning_rate = learning_rate;
    srand(static_cast<unsigned>(time(0)));

    this->hidden = Neuron();
    this->output = Neuron();
}


