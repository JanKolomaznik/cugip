#pragma once

#include <cugip/detail/defines.hpp>
#include <type_traits>

namespace cugip {

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



}  // namespace cugip
