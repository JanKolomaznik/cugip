#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>

namespace cugip {

namespace detail {

template<typename TInputType, typename TOutputType>
struct threshold_mask_operator
{
	threshold_mask_operator(TInputType  aThreshold, 
		                TOutputType aMaskValue, 
		                TOutputType aNonMaskValue): mThreshold(aThreshold), mMaskValue(aMaskValue), mNonMaskValue(aNonMaskValue)
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID TOutputType
	operator()(const TInputType &aValue) const
	{
		return aValue < aThreshold ? mNonMaskValue : mMaskValue;
	}

	TInputType  mThreshold; 
	TOutputType mMaskValue;
	TOutputType mNonMaskValue;
};

}//namespace detail


template <typename TInView, typename TOutView>
void 
threshold_masking(TInView                       aInView, 
                  TOutView                      aOutView, 
                  typename TInView::value_type  aThreshold, 
		  typename TOutView::value_type aMaskValue, 
		  typename TOutView::value_type aNonMaskValue = 0)
{
	typedef threshold_mask_operator<typename TInView::value_type, typename TOutView::value_type> threshold_op;
	cugip::transform(
			aInView, 
			aOutView, 
			threshold_op(aThreshold, aMaskValue, aNonMaskValue)
			);
}

}//namespace cugip


