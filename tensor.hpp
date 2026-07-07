
#ifndef TENSOR_HPP

#define TENSOR_HPP

#include<array>
#include<algorithm>
#include<cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <type_traits>

namespace ff {

namespace core {

    constexpr std::size_t max_dims = 4;
    template<typename dtype, size_t... shape_>
    struct Tensor;


    template<typename T>
    constexpr bool valid_index_type =
        std::is_integral_v<T> &&
        !std::is_same_v<std::__remove_cvref_t<T>, bool> &&
        !std::is_same_v<std::__remove_cvref_t<T>, char> &&
        !std::is_same_v<std::__remove_cvref_t<T>, signed char> &&
        !std::is_same_v<std::__remove_cvref_t<T>, unsigned char>;




    // Dependency for the tensor (for autograd)
    template<typename dtype, size_t... shape_>
    struct Dependency{
    Tensor<dtype, shape_...> *dependant_tensor;
    std::function<
            Tensor<dtype, shape_...>(Tensor<dtype, shape_...>)> assosiated_grad_fcn;
    };

    // Main Tensor struct 
    template<typename dtype, size_t... shape>
    struct Tensor{
private:
        // Number of dimensions (fixed at compile time)
        static constexpr size_t n_dim_ = []{
            constexpr size_t tmp_n_dims = sizeof...(shape);
            constexpr std::array<size_t, tmp_n_dims> temp_{shape...};

            size_t actual_dims = tmp_n_dims;
            for(size_t i = 0; i < tmp_n_dims; ++i){
                if(temp_[i] == 1){--actual_dims;}
            }
            return std::max(actual_dims, size_t{0});
        }();
        static_assert(n_dim_<= max_dims, "Max dimensions allowed: 4" );

        // Number of elements (fixed at compile time)
        static constexpr size_t num_elems_ = std::max((1*...*shape), size_t{1});

        // The tensor data storage
        std::array<dtype, num_elems_> data_ = {};

public:
        static constexpr size_t size = num_elems_;

        static constexpr std::array<size_t, max_dims> shape_ = []{
            std::array<size_t, max_dims> shape_arr = {};

            constexpr size_t tmp_n_dims = sizeof...(shape);
            constexpr std::array<size_t, tmp_n_dims> temp_ = {shape...};


            for(size_t i = 0; i < tmp_n_dims; ++i) shape_arr[max_dims - tmp_n_dims + i] = temp_[i];
            for(size_t j = 0; j < max_dims; j++) {
                if (shape_arr[j] == 0) {shape_arr[j] = 1;}
            }

            return shape_arr;
        }();

        // Lmabda for getting stride
        static constexpr std::array<size_t, max_dims> stride_ = []{
            std::array<size_t, max_dims> tmp_arr = {};
                if(n_dim_ > 0){
                    tmp_arr[max_dims - 1] = size_t{1};
                    for(size_t i = max_dims-2; i >= max_dims - n_dim_; --i) tmp_arr[i] = tmp_arr[i+1] * shape_[i+1];
                }

            return tmp_arr;
        }();

        // Constructor
        Tensor(std::array<dtype, num_elems_> const &data)
        :data_(data)
        {std::cout<<" "<<std::endl;std::cout<<n_dim_<<std::endl;}

        // direct indexing to internal data array. Hepls in broadcasting operations
        dtype at_flat(size_t idx){ 
            assert(idx < num_elems_);
            return data_[idx];}

        void assign_at_flat(size_t idx, dtype element){
            assert(idx < num_elems_);
            data_[idx] = element;
        }

        template<typename Idx0, typename... Index_types> 
        requires (
            (valid_index_type<Idx0>) &&
            ((valid_index_type<Index_types>) && ...))

        const dtype& operator()(Idx0 idx0,Index_types... indices){

            static_assert(sizeof...(indices) == n_dim_-1, "Indices used to access a tensor must equal its rank.");
            
            size_t idx_val = 0;
            if(n_dim_ == 1) {
                assert(idx0 < shape_[max_dims - 1]);
                idx_val = idx0;
            }
            
            else{
                std::array<size_t, n_dim_> idx_arr{size_t(idx0), size_t((indices)...)};

                for(size_t i = 0; i < n_dim_; i++) {
                    assert(idx_arr[i] >= 0 and idx_arr[i] < shape_[max_dims-n_dim_+i]);
                    idx_val += idx_arr[i]*stride_[max_dims-n_dim_+i];
                }
            }
            return data_[idx_val];
        }
    };


}
namespace tensor_op {

    template<typename dtype, size_t... shape_ret, size_t... shape_ten1, size_t... shape_ten2>
    bool add_tensor(core::Tensor<dtype, shape_ret...>* RET, 
                    core::Tensor<dtype, shape_ten1...>* ten1, 
                    core::Tensor<dtype, shape_ten2...>* ten2){
                        if (RET->size != std::max(ten1->size, ten2->size)){return false;}
                        for (size_t j = 0; j < core::max_dims; j ++) {
                            if(RET->shape_[j] != std::max(ten1->shape_[j], ten2->shape_[j])){return false;}
                        }
                        for(size_t i = 0; i < RET->shape_[0]; ++i){
                            for(size_t j = 0; j < RET->shape_[1]; ++j){
                                for(size_t k = 0; k < RET->shape_[2]; ++k){
                                    for(size_t l = 0; l < RET->shape_[3]; ++l){
                                        dtype element = 
                                            ten1->at_flat(ten1->stride_[0]*i + ten1->stride_[1]*j + ten1->stride_[2]*k + ten1->stride_[3]*l)
                                        +   ten2->at_flat(ten2->stride_[0]*i + ten2->stride_[1]*j + ten2->stride_[2]*k + ten2->stride_[3]*l);

                                        std::cout<<i<<j<<k<<l;
                                        std::cout<<element<<std::endl;

                                        RET->assign_at_flat(RET->stride_[0]*i + RET->stride_[1]*j + RET->stride_[2]*k + RET->stride_[3]*l, element);
                                    }
                                }
                            }
                        }

                        return true;
                    }

    template<typename dtype, size_t... shape_ret, size_t... shape_ten1, size_t... shape_ten2>
    bool subdtract_tensor(core::Tensor<dtype, shape_ret...>* RET, 
                    core::Tensor<dtype, shape_ten1...>* ten1, 
                    core::Tensor<dtype, shape_ten2...>* ten2){
                        if (RET->size != std::max(ten1->size, ten2->size)){return false;}
                        for (size_t j = 0; j < core::max_dims; j ++) {
                            if(RET->shape_[j] != std::max(ten1->shape_[j], ten2->shape_[j])){return false;}
                        }
                        for(size_t i = 0; i < RET->shape_[0]; ++i){
                            for(size_t j = 0; j < RET->shape_[1]; ++j){
                                for(size_t k = 0; k < RET->shape_[2]; ++k){
                                    for(size_t l = 0; l < RET->shape_[3]; ++l){
                                        dtype element = 
                                            ten1->at_flat(ten1->stride_[0]*i + ten1->stride_[1]*j + ten1->stride_[2]*k + ten1->stride_[3]*l)
                                        -   ten2->at_flat(ten2->stride_[0]*i + ten2->stride_[1]*j + ten2->stride_[2]*k + ten2->stride_[3]*l);

                                        std::cout<<i<<j<<k<<l;
                                        std::cout<<element<<std::endl;

                                        RET->assign_at_flat(RET->stride_[0]*i + RET->stride_[1]*j + RET->stride_[2]*k + RET->stride_[3]*l, element);
                                    }
                                }
                            }
                        }

                        return true;
                    }

    template<typename dtype, size_t... shape_ret, size_t... shape_ten1, size_t... shape_ten2>
    bool mult_tensor(core::Tensor<dtype, shape_ret...>* RET, 
                    core::Tensor<dtype, shape_ten1...>* ten1, 
                    core::Tensor<dtype, shape_ten2...>* ten2){
                        if (RET->size != std::max(ten1->size, ten2->size)){return false;}
                        for (size_t j = 0; j < core::max_dims; j ++) {
                            if(RET->shape_[j] != std::max(ten1->shape_[j], ten2->shape_[j])){return false;}
                        }
                        for(size_t i = 0; i < RET->shape_[0]; ++i){
                            for(size_t j = 0; j < RET->shape_[1]; ++j){
                                for(size_t k = 0; k < RET->shape_[2]; ++k){
                                    for(size_t l = 0; l < RET->shape_[3]; ++l){
                                        dtype element = 
                                            ten1->at_flat(ten1->stride_[0]*i + ten1->stride_[1]*j + ten1->stride_[2]*k + ten1->stride_[3]*l)
                                        *   ten2->at_flat(ten2->stride_[0]*i + ten2->stride_[1]*j + ten2->stride_[2]*k + ten2->stride_[3]*l);

                                        std::cout<<i<<j<<k<<l;
                                        std::cout<<element<<std::endl;

                                        RET->assign_at_flat(RET->stride_[0]*i + RET->stride_[1]*j + RET->stride_[2]*k + RET->stride_[3]*l, element);
                                    }
                                }
                            }
                        }

                        return true;
                    }
                }

}

#endif