#include "../tensor.hpp"
#include <iostream>
#include <ostream>


int main()
{
    ff::core::Tensor<float, 2, 2> a_ten({1.0, 2.0, 3.0, 4.0});
    ff::core::Tensor<float, 2, 2> b_ten({2.0, 3.0, 4.0, 5.0});
    ff::core::Tensor<float, 1, 2> c_ten({10.0, 15.0});

    ff::core::Tensor<float, 2, 2> ret_ten({0.0, 0.0, 0.0, 0.0});

    for(int i = 0; i < ff::core::max_dims; i ++){
        std::cout<<c_ten.stride_[i];}
    std::cout<<" "<<std::endl;
    
    if (ff::tensor_op::add_tensor(&ret_ten, &a_ten, &c_ten)){
        for(size_t i = 0; i < 4; i++) std::cout<<ret_ten.at_flat(i)<<std::endl;
    }

    float b = (float)a_ten(0, 1);

    std::cout<<b<<std::endl;

    return 0;
}