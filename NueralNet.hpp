#include<iostream>
#include<array>
#include<algorithm>
#include<cassert>


#include"Tensor.hpp"

class _lossFunction{};

class ActivationFunction{};

template<ActivationFunction actfun, unsigned int... num_layers>
class NeuralNet{
private:
    unsigned int num_hidden_layers;

public:
    NeuralNet()
    {
        assert(sizeof...(num_layers) > 2);

    }
};