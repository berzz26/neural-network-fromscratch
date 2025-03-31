#include <bits/stdc++.h>
using namespace std;

double randomWeight()
{
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x)); // Fixed parentheses
}

double sigmoidDerivative(double x)
{
    return x * (1 - x); // x should already be sigmoid(x)
}

class NeuralNetwork
{
public:
    vector<double> weights1, weights2;
    double bias1_h1, bias1_h2, bias2;
    double learningRate = 0.1;

    NeuralNetwork()
    {
        weights1 = {randomWeight(), randomWeight(), randomWeight(), randomWeight()};
        weights2 = {randomWeight(), randomWeight()};
        bias1_h1 = randomWeight();
        bias1_h2 = randomWeight();
        bias2 = randomWeight();
    }

    double forward(double x1, double x2)
    {
        double h1 = sigmoid(x1 * weights1[0] + x2 * weights1[1] + bias1_h1);
        double h2 = sigmoid(x1 * weights1[2] + x2 * weights1[3] + bias1_h2);
        double output = sigmoid(h1 * weights2[0] + h2 * weights2[1] + bias2);
        return output;
    }
};

void train(NeuralNetwork &nn, vector<vector<double>> X, vector<double> Y, int epochs)
{
    for (int i = 0; i < epochs; i++)
    {
        double totalError = 0;

        for (int j = 0; j < X.size(); j++)
        {
            double x1 = X[j][0], x2 = X[j][1];
            double expected = Y[j];

            // Forward Pass
            double h1 = sigmoid(x1 * nn.weights1[0] + x2 * nn.weights1[1] + nn.bias1_h1);
            double h2 = sigmoid(x1 * nn.weights1[2] + x2 * nn.weights1[3] + nn.bias1_h2);
            double output = sigmoid(h1 * nn.weights2[0] + h2 * nn.weights2[1] + nn.bias2);

            // Compute Error
            double error = expected - output;
            totalError += error * error;

            // Backpropagation
            double d_output = error * sigmoidDerivative(output);
            double d_h1 = d_output * nn.weights2[0] * sigmoidDerivative(h1);
            double d_h2 = d_output * nn.weights2[1] * sigmoidDerivative(h2);

            // Update Weights & Biases
            nn.weights2[0] += nn.learningRate * d_output * h1;
            nn.weights2[1] += nn.learningRate * d_output * h2;
            nn.bias2 += nn.learningRate * d_output;

            nn.weights1[0] += nn.learningRate * d_h1 * x1;
            nn.weights1[1] += nn.learningRate * d_h1 * x2;
            nn.weights1[2] += nn.learningRate * d_h2 * x1;
            nn.weights1[3] += nn.learningRate * d_h2 * x2;

            nn.bias1_h1 += nn.learningRate * d_h1;
            nn.bias1_h2 += nn.learningRate * d_h2;
        }

        if (i % 1000 == 0)
        {
            cout << "Epoch " << i << ", Error: " << totalError << endl;
        }
    }
}

int main()
{
    srand(time(0));

    NeuralNetwork nn;
    vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> Y = {0, 1, 1, 0}; // XOR outputs

    train(nn, X, Y, 10000); // 10k epochs

    cout << "Predictions:\n";
    for (auto &input : X)
    {
        cout << "Input: " << input[0] << ", " << input[1]
             << " -> Output: " << nn.forward(input[0], input[1]) << endl;
    }

    return 0;
}
