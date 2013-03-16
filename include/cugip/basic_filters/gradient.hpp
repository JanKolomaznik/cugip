#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/filter.hpp>

namespace cugip {
namespace detail {

template<typename TOutputType, typename TLocator, size_t tDim>
struct compute_symmetric_difference
{
	static CUGIP_DECL_HYBRID void 
	compute(TOutputType &aOutput, TLocator &aLocator)
	{
		get<tDim-1>(aOutput) = (aLocator.dim_offset<tDim-1>(1) - aLocator.dim_offset<tDim-1>(-1))/2;
		compute_symmetric_difference<TOutputType, TLocator, tDim - 1>::compute(aOutput, aLocator);
	}
};

//Stop the compile time recursion
template<typename TOutputType, typename TLocator>
struct compute_symmetric_difference<TOutputType, TLocator, 0>
{
	static CUGIP_DECL_HYBRID void 
	compute(TOutputType &, TLocator &)
	{ }
};

} //namespace detail


template<typename TInputType, typename TOutputType>
struct gradient_symmentric_difference
{
	template<typename TLocator>
	CUGIP_DECL_HYBRID TOutputType
	operator()(TLocator aLocator) const
	{
		TOutputType tmp;
		detail::compute_symmetric_difference<TOutputType, TLocator, dimension<TLocator>::value >::compute(tmp, aLocator);

		return tmp;
	}
};

}//namespace cugip
