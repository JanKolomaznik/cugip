#pragma once


#include <thrust/device_vector.hpp>
#include <thrust/host_vector.hpp>


template<typename TType>
view(thrust::device_vector<TType> &aVector) {

}

template<typename TType>
const_view(const thrust::device_vector<TType> &aVector) {

}
