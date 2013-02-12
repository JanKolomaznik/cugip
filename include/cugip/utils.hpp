#pragma once

#include <cugip/detail/include.hpp>

#if defined(__CUDACC__)
	#define CUGIL_DECL_HOST __host__
	#define CUGIL_DECL_DEVICE __device__
	#define CUGIL_DECL_HYBRID CUGIL_DECL_HOST CUGIL_DECL_DEVICE
	#define CUGIL_GLOBAL __global__
	#define CUGIL_CONSTANT __constant__
	#define CUGIL_SHARED __shared__
#else
	#define CUGIL_DECL_HOST
	#define CUGIL_DECL_DEVICE 
	#define CUGIL_DECL_HYBRID
	#define CUGIL_GLOBAL
	#define CUGIL_CONSTANT
	#define CUGIL_SHARED
#endif

#define CUGIL_ASSERT(...)

namespace cugip {


template<typename TType, int tChannelCount>
struct element
{
	TType data[tChannelCount];
};

typedef element<unsigned char, 3> element_rgb8_t;



}//namespace cugip
