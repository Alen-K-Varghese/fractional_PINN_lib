
#ifndef TENSOR_HPP

#define TENSOR_HPP

#include<array>
#include<algorithm>
#include<cassert>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace ft {
    template<typename T>
    constexpr bool valid_index_type =
        std::is_integral_v<T> &&
        !std::is_same_v<std::__remove_cvref_t<T>, bool> &&
        !std::is_same_v<std::__remove_cvref_t<T>, char> &&
        !std::is_same_v<std::__remove_cvref_t<T>, signed char> &&
        !std::is_same_v<std::__remove_cvref_t<T>, unsigned char>;


    template<typename dtype, size_t... shape>
    struct Tensor;

    // Dependency for the tensor (for autograd)
    template<typename dtype, size_t... shape>
    struct Dependency{
    Tensor<dtype, shape...> *dependant_tensor;
    std::function<
            Tensor<dtype, shape...>(Tensor<dtype, shape...>)> assosiated_grad_fcn;
    };

    // Main Tensor struct 
    template<typename dtype, size_t... shape>
    struct Tensor{
        // Number of dimensions (fixed at compile time)
        static constexpr size_t n_dim_ = sizeof...(shape);

        // Number of elements (fixed at compile time)
        static constexpr size_t num_elems_ = std::max((1*...*shape), size_t{1});

        static_assert(n_dim_<=4, "Max dimensions allowed:3(Rank 3 Tensor: Triad)");

        // Lambda for getting the shape an an array of size 4. Useful later for stride 
        // declaration and operations requiring broadcasting.
        static constexpr std::array<size_t, 4> shape_ = []{
            std::array<size_t, 4> tmp_arr{1, 1, 1, 1};
            constexpr std::array<size_t, n_dim_> temp_{shape...};
            for(size_t i = 0; i < n_dim_; ++i) tmp_arr[4-n_dim_+i] = temp_[i];

            return tmp_arr;
        }();

        // Lmabda for getting stride
        static constexpr std::array<size_t, 4> stride_ = []{
            std::array<size_t, 4> tmp_arr{1, 1, 1, 1};
            for(size_t i = 3; i-- > 0; ) tmp_arr[i] = tmp_arr[i+1] * std::max(shape_[i+1], size_t(1));

            return tmp_arr;
        }();

        // The tensor data storage
        std::array<dtype, num_elems_> data_ = {};

        bool requires_grad_ = false;

        // Constructor
        Tensor(std::array<dtype, num_elems_> const &data)
        :data_(data)
        {}

        template<typename Idx0, typename... Index_types> 
        requires (
            (valid_index_type<Idx0>) &&
            ((valid_index_type<Index_types>) && ...))

        const dtype& operator()(Idx0 idx0,Index_types... indices){

            static_assert(sizeof...(indices) == n_dim_-1, "Indices used to access a tensor must equal its rank.");
            
            size_t idx_val = 0;
            if(n_dim_ == 1) {
                assert(idx0 < shape_[3]);
                idx_val = idx0;
            }
            
            else{
                std::array<size_t, n_dim_> idx_arr{size_t(idx0), size_t((indices)...)};
                for(size_t i = 0; i < n_dim_; i++) {
                    assert(idx_arr[i] < shape_[4-n_dim_+i]);
                    idx_val += idx_arr[i]*stride_[4-n_dim_+i];
                }
            }
            return data_[idx_val];
        }
    };


}



#endif