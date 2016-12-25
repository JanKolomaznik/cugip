#pragma once

#include <cugip/detail/defines.hpp>
#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>


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

CUGIP_DECL_HOST inline std::ostream &
operator<<( std::ostream &stream, const dim3 &v )
{
	return stream << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
}


#endif //__CUDACC__

//TODO - move generic type traits to special header
template<int tIdx, typename TType>
struct get_policy;

template<int tIdx, typename TType>
CUGIP_DECL_HYBRID typename get_policy<tIdx, typename std::remove_reference<TType>::type >::return_type
get(TType &aArg)
{
	return get_policy<tIdx,
			  typename std::remove_reference<TType>::type
			  >::get(aArg);
}

template<int tIdx1, int tIdx2, typename TType>
struct get_policy2;


template<int tIdx1, int tIdx2, typename TType>
CUGIP_DECL_HYBRID float
get(TType &aArg)
{
	return get_policy2<tIdx1,
			  tIdx2,
			  typename std::remove_reference<TType>::type
			  >::get(aArg);
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

#ifdef __CUDACC__

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
