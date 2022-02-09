#pragma once

#include <cugip/detail/defines.hpp>
#include <cugip/cuda_error_check.hpp>
#include <cugip/detail/get_policy.hpp>
#include <cugip/detail/include.hpp>


namespace cugip {

//TODO - better organisation
#ifdef __CUDACC__

inline std::string
cudaMemoryInfoText()
{
	size_t free;
	size_t total;
	CUGIP_CHECK_RESULT(cudaMemGetInfo( &free, &total));

	return boost::str( boost::format("Free GPU memory: %1% MB; Total GPU memory %2% MB; Occupied %3%%%")
		% (float(free) / (1024*1024))
		% (float(total) / (1024*1024))
		% (100.0f * float(total - free)/total)
		);
}

inline std::string
cudaDeviceInfoText()
{
	std::string result;
	int count = 0;
	CUGIP_CHECK_RESULT(cudaGetDeviceCount(&count));

	result = boost::str(boost::format("Number of detected CUDA devices: %1%\n\n") % count);

	for (int i = 0; i < count; ++i) {
		cudaDeviceProp properties;
		CUGIP_CHECK_RESULT(cudaGetDeviceProperties(&properties, i));

		result += boost::str(boost::format("Name: %1%\nCompute capability: %2%.%3%\n\n")
				% properties.name
				% properties.major
				% properties.minor);
	}

	return result;
}

#endif //__CUDACC__

#ifdef __CUDACC__

template<size_t tIdx>
CUGIP_DECL_HYBRID unsigned int &
get(dim3 &aArg)
{
	switch (tIdx) {
	case 0:
		return aArg.x;
	case 1:
		return aArg.y;
	case 2:
		return aArg.z;
	}
	CUGIP_ASSERT(false);
	return aArg.x;
}

//*****************************************************************
//Extensions for built-in types
/*template<typename TType>
CUGIP_DECL_HOST void
swap(TType &aArg1, TType &aArg2)
{
	TType tmp = aArg1;
	aArg1 = aArg2;
	aArg2 = tmp;
}*/

/** \addtogroup auxiliary_function
 *  Auxiliary functions
 **/


CUGIP_DECL_DEVICE inline float
atomicFloatCAS(float *address, float old, float val)
{
	int i_val = __float_as_int(val);
	int tmp0 = __float_as_int(old);

	return __int_as_float(atomicCAS((int *)address, tmp0, i_val));
}

#endif //__CUDACC__

template<template<class> class TBoolTrait, typename THead, typename ...TTail>
struct fold_and {
	static constexpr bool value = TBoolTrait<THead>::value && fold_and<TBoolTrait, TTail...>::value;
};

template<template<class> class TBoolTrait, typename THead>
struct fold_and<TBoolTrait, THead> {
	static constexpr bool value = TBoolTrait<THead>::value;
};


}//namespace cugip
