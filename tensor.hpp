
#ifndef TENSOR_HPP

#define TENSOR_HPP

#include<array>
#include<algorithm>
#include<cassert>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace ff {

namespace core {

    constexpr std::size_t max_dims = 4;

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
private:
        // Number of dimensions (fixed at compile time)
        static constexpr size_t n_dim_ = sizeof...(shape);

        // Number of elements (fixed at compile time)
        static constexpr size_t num_elems_ = std::max((1*...*shape), size_t{1});

        static_assert(n_dim_<= max_dims, "Max dimensions allowed: 4" );


        static constexpr std::array<size_t, max_dims> shape_ = []{
            std::array<size_t, max_dims> shape_arr = {};
            constexpr std::array<size_t, n_dim_> temp_{shape...};

            for(size_t i = 0; i < n_dim_; ++i) shape_arr[max_dims - n_dim_ + i] = temp_[i];
            for(size_t j = 0; j < max_dims; j++) {
                if (shape_arr[j] == 0) shape_arr[j] = 1;
            }

            return shape_arr;
        }();

        // Lmabda for getting stride
        static constexpr std::array<size_t, max_dims> stride_ = []{
            std::array<size_t, max_dims> tmp_arr = {};
            tmp_arr[max_dims - 1] = size_t{1};
            for(size_t i = max_dims-2; i >= max_dims - n_dim_; --i) tmp_arr[i] = tmp_arr[i+1] * shape_[i+1];

            return tmp_arr;
        }();

        // The tensor data storage
        std::array<dtype, num_elems_> data_ = {};

        // Declaration of helper functions as friend
        template<typename dtype_ret, size_t... shape_ret,
                typename dtype_ten1, size_t... shape_ten1,
                typename dtype_ten2, size_t... shape_ten2>
        
        friend bool add_tensor(core::Tensor<dtype_ret, shape_ret...>* RET, 
                        core::Tensor<dtype_ten1, shape_ten1...>* ten1, 
                        core::Tensor<dtype_ten2, shape_ten2...>* ten2);
        template<typename dtype_ret, size_t... shape_ret,
                typename dtype_ten1, size_t... shape_ten1,
                typename dtype_ten2, size_t... shape_ten2>
        
        bool substract_tensor(core::Tensor<dtype_ret, shape_ret...>* RET, 
                        core::Tensor<dtype_ten1, shape_ten1...>* ten1, 
                        core::Tensor<dtype_ten2, shape_ten1...>* ten2);

        template<typename dtype_ret, size_t... shape_ret,
                typename dtype_ten1, size_t... shape_ten1,
                typename dtype_ten2, size_t... shape_ten2>
        
        bool mult_tensor(core::Tensor<dtype_ret, shape_ret...>* RET, 
                        core::Tensor<dtype_ten1, shape_ten1...>* ten1, 
                        core::Tensor<dtype_ten2, shape_ten1...>* ten2);
public:
        // Constructor
        Tensor(std::array<dtype, num_elems_> const &data)
        :data_(data)
        {}

        // Getting the shape of the tensor
        std::array<dtype, n_dim_> get_shape(){ return shape_; }

        // Getting the number of elements in the tensor
        size_t get_size(){return num_elems_;}

        // direct indexing to internal data array. Hepls in broadcasting operations
        dtype at_flat(size_t idx){ 
            assert(idx < num_elems_);
            return data_[idx];}

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
    template<typename dtype_ret, size_t... shape_ret,
            typename dtype_ten1, size_t... shape_ten1,
            typename dtype_ten2, size_t... shape_ten2>
    
    bool add_tensor(core::Tensor<dtype_ret, shape_ret...>* RET, 
                    core::Tensor<dtype_ten1, shape_ten1...>* ten1, 
                    core::Tensor<dtype_ten2, shape_ten2...>* ten2){
                        assert(RET->num_elems_ == std::max(ten1->num_elems_, ten2->num_elems_));


                    }

    template<typename dtype_ret, size_t... shape_ret,
            typename dtype_ten1, size_t... shape_ten1,
            typename dtype_ten2, size_t... shape_ten2>
    
    bool substract_tensor(core::Tensor<dtype_ret, shape_ret...>* RET, 
                    core::Tensor<dtype_ten1, shape_ten1...>* ten1, 
                    core::Tensor<dtype_ten2, shape_ten1...>* ten2){
                        assert(RET->num_elems_ == std::max(ten1->num_elems_, ten2->num_elems_));

                    }

    template<typename dtype_ret, size_t... shape_ret,
            typename dtype_ten1, size_t... shape_ten1,
            typename dtype_ten2, size_t... shape_ten2>
    
    bool add_tensor(core::Tensor<dtype_ret, shape_ret...>* RET, 
                    core::Tensor<dtype_ten1, shape_ten1...>* ten1, 
                    core::Tensor<dtype_ten2, shape_ten1...>* ten2){
                        assert(RET->num_elems_ == std::max(ten1->num_elems_, ten2->num_elems_));
                    
                    }
}

#endif