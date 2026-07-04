#include "../tensor.hpp"
#include <iostream>

int main()
{
    ft::Tensor<float, 2, 2> a_ten({1.0, 2.0, 3.0, 4.0});

    float b = (float)a_ten(1, 1);

    std::cout<<b<<std::endl;

    return 0;
}