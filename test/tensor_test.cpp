#include "../tensor.hpp"
#include <iostream>

int main()
{
    ff::core::Tensor<float, 2, 2> a_ten({1.0, 2.0, 3.0, 4.0});

    float b = (float)a_ten(0, 4);

    std::cout<<b<<std::endl;
    std::cout<<a_ten.n_dim_<<std::endl;

    return 0;
}