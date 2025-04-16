#include "Network.hpp"
#include <iostream>
#include <ctime>
#include <iostream>

Layer::Layer(int n_neurons, int n_past_neurons)
{
    this->b = this->b = rand() / RAND_MAX * 2.0f - 1.0f;
    this->n_neurons = n_neurons;
    this->n_past_neurons = n_past_neurons;

    this->weights.resize(static_cast<size_t>(this->n_neurons * this->n_past_neurons));
    for(int i = 0;i < weights.size();i++)
        weights[i] = this->b = rand() / RAND_MAX * 2.0f - 1.0f;
}

std::vector<float> Layer::foward(std::vector<float> x)
{
    std::vector<float> output;
    output.resize(static_cast<size_t>(this->n_neurons));

    for(int n = 0;n < this->n_neurons;n++)
    {
        for(int v = 0;v < x.size();v++)
        {
            for(int w = 0;w < this->n_past_neurons;w++)
            {
                output[n] = this->weights[w] * x[v] + this->b;
            }
        }
    }

    return output;
}

std::vector<float> Layer::back(float delta, std::vector<float> past_weights)
{
    std::vector<float> output;
    output.resize(past_weights.size());

    for(int n = 0;n < output.size();n++)
    {
        output[n] = past_weights[n] * delta;
    }

    return output;
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

void Network::train(std::vector<std::pair<float,float>> input_data, std::vector<float> predict_data, const int &epochs)
{
    for(int e = 0;e < epochs;e++){
        float actual_loss = 0.0f;
        for (int i = 0;i < input_data.size();i++)
        {
            float pred = this->predict(input_data[i].first, input_data[i].second);
            float loss = mse(predict_data[i], pred);
            actual_loss += loss;
            
            std::vector<float> inputs = {input_data[i].first, input_data[i].second};
            backpropragation(inputs, pred, predict_data[i]);
        }
        std::cout << "Epoch: " << e << " loss: " << actual_loss << "\n";
    }
}

float Network::predict(const float &x, const float& y)
{
    std::vector<float> preds = hidden.foward({x, y});
    preds = out.foward(preds);
    float pred_transfomed = 0.0f;
    for (float& pred : preds)
        pred_transfomed += sigmoid(pred);
    return pred_transfomed;
}

void Network::backpropragation(std::vector<float> inputs, const float& pred, const float& real_pred)
{
    std::vector<float> output1 = hidden.foward(inputs);
    out.foward(output1);

    float erro = pred - real_pred;
    float delta2 = erro * sigmoid_derivate(pred);

    std::vector<float> wo = out.back(delta2, output1);

    float delta1 = delta2 * out.weights[0];

    std::vector<float> wh = hidden.back(delta1, inputs);

    for(int i = 0;i < hidden.weights.size();i++)
        hidden.weights[i] -= this->learning_rate * wh[i];

    for(int i = 0;i < out.weights.size();i++)
        out.weights[i] -= this->learning_rate * wo[i];

    hidden.b -= this->learning_rate * delta1;
    out.b -= this->learning_rate * delta2;
}

Network::Network(float learning_rate)
{
    this->learning_rate = learning_rate;
    srand(static_cast<unsigned>(time(0)));

    hidden = Layer(1);
    out = Layer(1);
}
