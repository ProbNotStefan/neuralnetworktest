#include <bits/stdc++.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

MatrixXd sigmoid(const MatrixXd &z)
{
    return z.unaryExpr([](double val) { return 1.0 / (1.0 + exp(-val)); });
}

MatrixXd dsigmoid(const MatrixXd &z)
{
    return z.unaryExpr([](double val) { return val * (1.0 - val); });
}

// load mnist images and labels
MatrixXd ldMNISTI(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Probably wrong path to train-images.idx3-ubyte: " + filename);
    int magicn = 0, numI = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char *>(&magicn), sizeof(magicn));
    file.read(reinterpret_cast<char *>(&numI), sizeof(numI));
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

    magicn = __builtin_bswap32(magicn);
    numI = __builtin_bswap32(numI);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    MatrixXd images(rows * cols, numI);
    for (int i = 0; i < numI; ++i)
    {
        for (int j = 0; j < rows * cols; ++j)
        {
            unsigned char p = 0;
            file.read(reinterpret_cast<char *>(&p), sizeof(p));
            images(j, i) = static_cast<double>(p) / 255.0;
        }
    }
    file.close();
    return images;
}

MatrixXd ldMNISTl(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Probably wrong path to train-labels.idx1-ubyte: " + filename);

    int magicn = 0, numL = 0;
    file.read(reinterpret_cast<char *>(&magicn), sizeof(magicn));
    file.read(reinterpret_cast<char *>(&numL), sizeof(numL));
    magicn = __builtin_bswap32(magicn);
    numL = __builtin_bswap32(numL);
    MatrixXd labels(10, numL);
    labels.setZero();

    for (int i = 0; i < numL; ++i)
    {
        unsigned char l = 0;
        file.read(reinterpret_cast<char *>(&l), sizeof(l));
        labels(l, i) = 1.0;
    }

    file.close();
    return labels;
}

//
class NeuralNetwork
{
    vector<MatrixXd> weights;
    vector<MatrixXd> biases;
    double alpha;

    public:
        NeuralNetwork(const vector<int> &layers, double lr = 0.01) : alpha(lr)
        {
            for (size_t i = 1; i < layers.size(); ++i)
            {
                weights.push_back(MatrixXd::Random(layers[i], layers[i - 1]));
                biases.push_back(MatrixXd::Random(layers[i], 1));
            }
        }

        // easy part
        vector<MatrixXd> feedforward(const MatrixXd &input)
        {
            if (input.rows() != weights[0].cols())
            {
                throw runtime_error("Input/layer size mismatch.");
            }
            vector<MatrixXd> activations;
            activations.push_back(input);
            MatrixXd cur = input;
            for (size_t i = 0; i < weights.size(); ++i)
            {
                cur = sigmoid((weights[i] * cur).colwise() + biases[i].col(0));
                activations.push_back(cur);
            }
            return activations;
        }

        // chain thingy
        void backpropagate(const MatrixXd &input, const MatrixXd &output)
        {
            vector<MatrixXd> activations = feedforward(input);
            vector<MatrixXd> d(weights.size());
            d.back() = (activations.back() - output).cwiseProduct(dsigmoid(activations.back()));
            for (int i = weights.size() - 2; i >= 0; --i)
            {
                d[i] = (weights[i + 1].transpose() * d[i + 1]).cwiseProduct(dsigmoid(activations[i + 1]));
            }
            for (size_t i = 0; i < weights.size(); ++i)
            {
                weights[i] -= alpha * d[i] * activations[i].transpose();
                biases[i] -= alpha * d[i].rowwise().mean();
            }
        }

        void train(const vector<pair<MatrixXd, MatrixXd>> &data, int epochs, int bS)
        {
            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                double cost = 0.0;
                for (int batch = 0; batch < data.size() / bS; ++batch)
                {
                    MatrixXd bIn(data[0].first.rows(), bS);
                    MatrixXd y(data[0].second.rows(), bS);

                    for (int i = 0; i < bS; ++i)
                    {
                        bIn.col(i) = data[batch * bS + i].first;
                        y.col(i) = data[batch * bS + i].second;
                    }
                    vector<MatrixXd> activations = feedforward(bIn);
                    MatrixXd y_hat = activations.back();
                    cost += (y_hat - y).array().square().mean();
                    backpropagate(bIn, y);
                }
                cout << "epoch  " << epoch + 1 << " / " << epochs << " - cost: " << cost / (data.size() / bS) << endl;
            }
        }

        // pleut
        void saveModel(const string &filename)
        {
            ofstream file(filename, ios::binary);
            for (const auto &w : weights)
            {
                file.write(reinterpret_cast<const char *>(w.data()), w.size() * sizeof(double));
            }
            for (const auto &b : biases)
            {
                file.write(reinterpret_cast<const char *>(b.data()), b.size() * sizeof(double));
            }
            file.close();
        }

        void loadModel(const string &filename)
        {
            ifstream file(filename, ios::binary);
            for (auto &w : weights)
            {
                file.read(reinterpret_cast<char *>(w.data()), w.size() * sizeof(double));
            }
            for (auto &b : biases)
            {
                file.read(reinterpret_cast<char *>(b.data()), b.size() * sizeof(double));
            }
            file.close();
        }
};

int main()
{
    vector<int> layers = {784, 128, 64, 10};
    NeuralNetwork nn(layers, 0.01);
    string imagesFile = "C:\\Users\\djuma\\OneDrive\\Documents\\CS\\DEV\\C++\\AI\\train-images.idx3-ubyte";
    string labelsFile = "C:\\Users\\djuma\\OneDrive\\Documents\\CS\\DEV\\C++\\AI\\train-labels.idx1-ubyte";
    MatrixXd images = ldMNISTI(imagesFile);
    MatrixXd labels = ldMNISTl(labelsFile);

    vector<pair<MatrixXd, MatrixXd>> dataset;
    for (int i = 0; i < images.cols(); ++i)
    {
        dataset.push_back({images.col(i), labels.col(i)});
    }

    nn.train(dataset, 10, 16);
    nn.saveModel("C:\\Users\\djuma\\OneDrive\\Documents\\CS\\DEV\\C++\\AI\\model.dat");
    return 0;
}
