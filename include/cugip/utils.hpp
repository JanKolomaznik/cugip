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

template<typename TType, int tChannelCount>
struct element
{
	typedef TType channel_t;
	static const size_t dimension;

	TType data[tChannelCount];

	CUGIP_DECL_HYBRID element &
	operator=(const element &aArg)
	{
		for (size_t i = 0; i < tChannelCount; ++i) {
			data[i] = aArg.data[i];
		}
		return *this;
	}

	CUGIP_DECL_HYBRID element &
	operator=(const TType &aArg)
	{
		for (size_t i = 0; i < tChannelCount; ++i) {
			data[i] = aArg;
		}
		return *this;
	}
};

template<size_t tIdx, typename TType>
struct get_policy;


/*template<size_t tIdx, typename TType, size_t tChannelCount>
struct get_policy<tIdx, const element<TType, tChannelCount> >
{
	typedef const TType & return_type;
	typedef const element<TType, tChannelCount> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		return aArg.data[tIdx];
	}
};

template<size_t tIdx, typename TType, size_t tChannelCount>
struct get_policy<tIdx, element<TType, tChannelCount> >
{
	typedef TType & return_type;
	typedef element<TType, tChannelCount> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		return aArg.data[tIdx];
	}

};*/

template<size_t tIdx, typename TType>
CUGIP_DECL_HYBRID typename get_policy<tIdx, typename boost::remove_reference<TType>::type >::return_type
get(TType &aArg)
{
	return get_policy<tIdx,
			  typename boost::remove_reference<TType>::type
			  >::get(aArg);
}

template<size_t tIdx, typename TType, int tChannelCount>
CUGIP_DECL_HYBRID TType &
get(element<TType, tChannelCount> &aArg)
{
	return aArg.data[tIdx];
}

template<size_t tIdx, typename TType, int tChannelCount>
CUGIP_DECL_HYBRID const TType &
get(const element<TType, tChannelCount> &aArg)
{
	//std::cout << typeid(aArg).name() << std::endl;
	return aArg.data[tIdx]; //get_policy<tIdx,
			  //const element<TType, tChannelCount>
			  //>::get(aArg);
}


typedef element<unsigned char, 3> element_rgb8_t;
//typedef element<unsigned char, 1> element_gray8_t;
typedef unsigned char element_gray8_t;
typedef char element_gray8s_t;
typedef element<signed char, 2> element_channel2_8s_t;


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

/**
 * \param aSharedBuffer Buffer of block size length in shared memory
 **/
template<typename TType>
CUGIP_DECL_DEVICE const TType &
block_prefix_sum(int aTid, int blockSize, const TType &aCurrent, TType *aSharedBuffer) {
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
	/*__syncthreads();
	if (aTid == 0) {
		aSharedBuffer[0] = 0;
	}
	__syncthreads();*/
	return aSharedBuffer[aTid];
}



template<typename TType>
CUGIP_DECL_DEVICE TType
block_prefix_sum2(int aTid, int blockSize, const TType &aCurrent, TType *aSharedBuffer) {
	__shared__ TType temp[32];
	TType temp1, temp2;
	//int tid = threadIdx.x;
	temp1 = aCurrent;//d_data[aTid+blockIdx.x*blockDim.x];
	for (int d=1; d<32; d<<=1) {
		temp2 = __shfl_up(temp1,d);
		if (aTid%32 >= d) temp1 += temp2;
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
	if (aTid == blockSize -1) {
		aSharedBuffer[blockSize] = temp1;
	}
	return temp1;
}

template<typename TType>
CUGIP_DECL_DEVICE TType
block_prefix_sum3(int aTid, int blockSize, const TType &aCurrent, TType *aSharedBuffer) {
	return block_prefix_sum2<TType>(aTid, blockSize, aCurrent, aSharedBuffer) - aCurrent;
}

}//namespace cugip
