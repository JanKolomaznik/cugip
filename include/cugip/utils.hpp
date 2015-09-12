#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>
#include <cugip/utils.hpp>

#if defined(__CUDACC__)
	#define CUGIP_DECL_HOST __host__
	#define CUGIP_DECL_DEVICE __device__
	#define CUGIP_DECL_HYBRID CUGIP_DECL_HOST CUGIP_DECL_DEVICE
	#define CUGIP_GLOBAL __global__
	#define CUGIP_CONSTANT __constant__
	#define CUGIP_SHARED __shared__

	// Disables "host inside device function warning"
	#define CUGIP_HD_WARNING_DISABLE #pragma hd_warning_disable
#else
	#define CUGIP_DECL_HOST
	#define CUGIP_DECL_DEVICE
	#define CUGIP_DECL_HYBRID
	#define CUGIP_GLOBAL
	#define CUGIP_CONSTANT
	#define CUGIP_SHARED

	#define CUGIP_HD_WARNING_DISABLE
#endif //__CUDACC__

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


//*****************************************************************
//Extensions for built-in types
CUGIP_DECL_HOST inline std::ostream &
operator<<( std::ostream &stream, const dim3 &v )
{
	return stream << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
}

template<typename TType>
CUGIP_DECL_HOST void
swap(TType &aArg1, TType &aArg2)
{
	TType tmp = aArg1;
	aArg1 = aArg2;
	aArg2 = tmp;
}

/** \defgroup auxiliary_function
 *
 **/


CUGIP_DECL_DEVICE inline float
atomicFloatCAS(float *address, float old, float val)
{
	int i_val = __float_as_int(val);
	int tmp0 = __float_as_int(old);

	return __int_as_float(atomicCAS((int *)address, tmp0, i_val));
}


template<typename TType>
struct ScanResult {
	TType current;
	TType total;
};

/**
 * \param aSharedBuffer Buffer of block size length in shared memory
 **/
template<typename TType>
CUGIP_DECL_DEVICE //const TType &
ScanResult<TType>
block_prefix_sum_in(int aTid, int blockSize, const TType &aCurrent, TType *aSharedBuffer)
{
//#if __CUDA_ARCH__ >= 300
	__shared__ TType temp[32];
	TType temp1, temp2;
	//int tid = threadIdx.x;
	temp1 = aCurrent;//d_data[aTid+blockIdx.x*blockDim.x];
	for (int d=1; d<32; d<<=1) {
		temp2 = __shfl_up(temp1,d);
		if (aTid%32 >= d) temp1 += temp2;
		__syncthreads();
	}
	if (aTid%32 == 31) temp[aTid/32] = temp1;
	__syncthreads();

	if (aTid < 32) {
		temp2 = 0.0f;
		if (aTid < blockSize/32)
			temp2 = temp[aTid];
		for (int d=1; d<32; d<<=1) {
			TType temp3 = __shfl_up(temp2,d);
			if (aTid%32 >= d) temp2 += temp3;
		}
		if (aTid < blockDim.x/32) temp[aTid] = temp2;
	}
	__syncthreads();

	if (aTid >= 32) temp1 += temp[aTid/32 - 1];
	__syncthreads();
	if (aTid == blockSize -1) {
		aSharedBuffer[blockSize] = temp1;
		temp[0] = temp1;
		//printf("Total ss %d\n", temp1);
	}
	__syncthreads();
	//return temp1;
	return ScanResult<TType>{ temp1, temp[0] };
/*#else
	TType sum = aCurrent;
	aSharedBuffer[aTid] = sum;
	if (aTid == 0) {
		aSharedBuffer[0] = 0;
	}
	__syncthreads();
	for(int offset = 1; offset < blockSize; offset <<= 1) {
		if(aTid >= offset) {
			sum += aSharedBuffer[aTid - offset];
		}

		// wait until every thread has updated its partial sum
		__syncthreads();

		// write my partial sum
		aSharedBuffer[aTid] = sum;

		// wait until every thread has written its partial sum
		__syncthreads();
	}
	return aSharedBuffer[aTid];
#endif*/
}

template<typename TType>
CUGIP_DECL_DEVICE //TType
ScanResult<TType>
block_prefix_sum_ex(int aTid, int blockSize, const TType &aCurrent, TType *aSharedBuffer) {
	ScanResult<TType> res = block_prefix_sum_in<TType>(aTid, blockSize, aCurrent, aSharedBuffer);
	return ScanResult<TType>{ res.current - aCurrent, res.total };

	/* Excluded
	TType sum = aCurrent;
	aSharedBuffer[aTid+1] = sum;
	if (aTid == 0) {
		aSharedBuffer[0] = 0;
	}
	__syncthreads();
	for(int offset = 1; offset < blockSize; offset <<= 1) {
		if(aTid >= offset) {
			sum += aSharedBuffer[aTid - offset + 1];
		}

		// wait until every thread has updated its partial sum
		__syncthreads();

		// write my partial sum
		aSharedBuffer[aTid + 1] = sum;

		// wait until every thread has written its partial sum
		__syncthreads();
	}
	return aSharedBuffer[aTid];*/

}





}//namespace cugip
