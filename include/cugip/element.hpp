#pragma once

#include <cugip/utils.hpp>
#include <cugip/math.hpp>

namespace cugip {

template<typename TType, int tChannelCount>
struct element
{
	typedef TType channel_t;
	static const int dimension;

	TType data[tChannelCount];

	inline CUGIP_DECL_HYBRID TType &
	operator[](int aIdx)
	{
		return data[aIdx];
	}

	inline CUGIP_DECL_HYBRID const TType &
	operator[](int aIdx)const
	{
		return data[aIdx];
	}

	CUGIP_DECL_HYBRID element &
	operator=(const element &aArg)
	{
		for (int i = 0; i < tChannelCount; ++i) {
			data[i] = aArg.data[i];
		}
		return *this;
	}

	CUGIP_DECL_HYBRID element &
	operator=(const TType &aArg)
	{
		for (int i = 0; i < tChannelCount; ++i) {
			data[i] = aArg;
		}
		return *this;
	}
};

template<typename TType, int tDim>
struct dimension<element<TType, tDim> >: dimension_helper<tDim> {};


/*template<int tIdx, typename TType, int tChannelCount>
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

template<int tIdx, typename TType, int tChannelCount>
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

template<int tIdx, typename TType, int tChannelCount>
CUGIP_DECL_HYBRID TType &
get(element<TType, tChannelCount> &aArg)
{
	return aArg.data[tIdx];
}

template<int tIdx, typename TType, int tChannelCount>
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


}//namespace cugip
