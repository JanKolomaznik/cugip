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
struct get_policy
{
	typedef const typename TType::channel_t const_value_t;
	typedef typename TType::channel_t value_t;

	static CUGIP_DECL_HYBRID value_t &
	get(typename boost::call_traits<TType>::reference aArg)
	{
		return aArg.data[tIdx];
	}

	static CUGIP_DECL_HYBRID const_value_t &
	const_get(typename boost::call_traits<TType>::const_reference aArg)
	{
		return aArg.data[tIdx];
	}
};

/*template<size_t tIdx, typename TType, size_t tChannelCount>
struct get_policy<tIdx, element<TType, tChannelCount> >
{
	typedef const TType const_value_t;
	typedef TType value_t;

	static CUGIP_DECL_HYBRID const_value_t &
	get(const element<TType, tChannelCount> &aArg)
	{
		return aArg.data[tIdx];
	}

	static CUGIP_DECL_HYBRID value_t &
	get(element<TType, tChannelCount> &aArg)
	{
		return aArg.data[tIdx];
	}
};*/

/*template<size_t tIdx, typename TType>
CUGIP_DECL_HYBRID typename TType::channel_t &
getl(TType &aArg)
{
	return  get_policy<tIdx, TType>::get(aArg);//aArg.data[tIdx];
}*/

template<size_t tIdx, typename TType>
CUGIP_DECL_HYBRID typename get_policy<tIdx, TType>::const_value_t &
get(const TType &aArg)
{
	return get_policy<tIdx, typename boost::remove_reference<TType>::type>::const_get(aArg);
}

template<size_t tIdx, typename TType>
CUGIP_DECL_HYBRID typename get_policy<tIdx, TType>::value_t &
get(TType &aArg)
{
	return get_policy<tIdx, typename boost::remove_reference<TType>::type>::get(aArg);
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


}//namespace cugip
