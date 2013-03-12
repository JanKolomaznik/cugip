#pragma once
#include <cugip/math.hpp>

namespace cugip {

template <typename TType, size_t tDim>
struct convolution_mask
{
	typename dim_traits<tDim>::coord_t mFrom;
	typename dim_traits<tDim>::coord_t mTo;
	
};

/** \ingroup  traits
 * @{
 **/
template <typename TType, size_t tDim>
struct dimension<convolution_mask<TType, tDim> >
{
	static const size_t value = tDim;
};
/** 
 * @}
 **/


/** \ingroup auxiliary_function
 * @{
 **/

template<typename TConvolutionMask, size_t tDim>
CUGIP_FORCE_INLINE int
from(const TConvolutionMask& aMask) 
{
	return get<tDim>(aMask.mFrom);
}

template<typename TConvolutionMask, size_t tDim>
CUGIP_FORCE_INLINE int
to(const TConvolutionMask& aMask) 
{
	return get<tDim>(aMask.mTo);
}

/** 
 * @}
 **/

	
namespace detail {

CUGIP_DECL_HYBRID

apply_convolution(const TConvolutionMask &aMask, TAccessor &aAccessor) {
	TOutputType tmp = 0;
	for (int i = from<0>(aMask); i < to<0>(aMask); ++i) {
		
	}

}

template<typename TInputType, typename TOutputType, typename TConvolutionMask>
struct convolution_operator
{
	convolution_operator(const TConvolutionMask&aMask): mask(aMask)
	{}

	template<typename TAccessor>
	CUGIP_DECL_HYBRID TOutputType
	operator()(TAccessor aAccessor) const
	{
		TOutputType tmp = 0;
		//tmp.data[0] = abs(aAccessor[typename TAccessor::diff_t(-1,0)].data[0] - aAccessor[typename TAccessor::diff_t()].data[0]) + abs(aAccessor[typename TAccessor::diff_t(0,-1)].data[0] - aAccessor[typename TAccessor::diff_t()].data[0]);
		return tmp;
	}

	TConvolutionMask mask;
};

}//namespace detail


template <typename TInView, typename TOutView, typename TConvolutionMask>
void 
convolution(TInView aInView, TOutView aOutView, TConvolutionMask aConvolutionMask)
{
	cugip::filter(
			aInView, 
			aOutView, 
			detail::convolution_operator<typename TInView::value_type, typename TOutView::value_type, TConvolutionMask>(aConvolutionMask)
			);
}

}//namespace cugip

