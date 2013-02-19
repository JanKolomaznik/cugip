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

#define CUGIL_ASSERT(EXPR) assert(EXPR)

#define CUGIL_ASSERT_RESULT(EXPR) CUGIL_ASSERT(cudaSuccess == EXPR)

#define CUGIL_CHECK_RESULT(EXPR) EXPR

namespace cugip {


inline std::string
cudaMemoryInfoText()
{
	size_t free;
	size_t total;
	CUGIL_CHECK_RESULT(cudaMemGetInfo( &free, &total));

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


//*****************************************************************
//Extensions for built-in types
CUGIL_DECL_HOST inline std::ostream &
operator<<( std::ostream &stream, const dim3 &v )
{
	return stream << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
}

}//namespace cugip
