#pragma once

#include <cstddef>

#define CUGIP_ASSERT(EXPR) assert(EXPR)

#if defined(CUGIP_DEBUG_ACCESS)
	#define CUGIP_ACCESS_ASSERT(EXPR) assert(EXPR)
#else
	#define CUGIP_ACCESS_ASSERT(EXPR)
#endif //CUGIP_DEBUG_ACCESS

#if defined(__CUDACC__)
	#define CUGIP_DECL_HOST __host__
	#define CUGIP_DECL_DEVICE __device__
	#define CUGIP_DECL_HYBRID CUGIP_DECL_HOST CUGIP_DECL_DEVICE
	#define CUGIP_GLOBAL __global__
	#define CUGIP_CONSTANT __constant__
	#define CUGIP_SHARED __shared__

	// Disables "host inside device function warning"
	#if defined(__CUDACC__) && defined(__NVCC__)
		#if __CUDAVER__ >= 75000
			#define CUGIP_HD_WARNING_DISABLE #pragma nv_exec_check_disable
		#else
			#define CUGIP_HD_WARNING_DISABLE #pragma hd_warning_disable
		#endif
	#else
		#define CUGIP_HD_WARNING_DISABLE
	#endif


	#define CUGIP_ASSERT_RESULT(EXPR) CUGIP_ASSERT(cudaSuccess == EXPR)
#else
	#define CUGIP_DECL_HOST
	#define CUGIP_DECL_DEVICE
	#define CUGIP_DECL_HYBRID
	#define CUGIP_GLOBAL
	#define CUGIP_CONSTANT
	#define CUGIP_SHARED

	#define CUGIP_HD_WARNING_DISABLE

	using cudaStream_t = size_t;
#endif //__CUDACC__

#define CUGIP_FORCE_INLINE inline
