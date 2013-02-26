#pragma once

namespace cugip {

namespace detail {

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

