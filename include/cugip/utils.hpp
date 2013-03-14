#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>

#if defined(__CUDACC__)
	#define CUGIP_DECL_HOST __host__
	#define CUGIP_DECL_DEVICE __device__
	#define CUGIP_DECL_HYBRID CUGIP_DECL_HOST CUGIP_DECL_DEVICE
	#define CUGIP_GLOBAL __global__
	#define CUGIP_CONSTANT __constant__
	#define CUGIP_SHARED __shared__
#else
	#define CUGIP_DECL_HOST
	#define CUGIP_DECL_DEVICE 
	#define CUGIP_DECL_HYBRID
	#define CUGIP_GLOBAL
	#define CUGIP_CONSTANT
	#define CUGIP_SHARED
#endif

#define CUGIP_ASSERT(EXPR) assert(EXPR)

#define CUGIP_ASSERT_RESULT(EXPR) CUGIP_ASSERT(cudaSuccess == EXPR)

#define CUGIP_FORCE_INLINE inline



namespace cugip {


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



template<typename TType, int tChannelCount>
struct element
{
	TType data[tChannelCount];
};

typedef element<unsigned char, 3> element_rgb8_t;
//typedef element<unsigned char, 1> element_gray8_t;
typedef unsigned char element_gray8_t;
typedef char element_gray8s_t;


//*****************************************************************
//Extensions for built-in types
CUGIP_DECL_HOST inline std::ostream &
operator<<( std::ostream &stream, const dim3 &v )
{
	return stream << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
}

/** \defgroup auxiliary_function
 * 
 **/


}//namespace cugip
