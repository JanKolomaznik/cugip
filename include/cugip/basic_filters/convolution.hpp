#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>

namespace cugip {

template <typename TType, typename TSizeTraits>
struct convolution_kernel
{
	typename dim_traits<TSizeTraits::dimension>::diff_t mOffset;

	TType data[TSizeTraits::size];
};

/** \ingroup  traits
 * @{
 **/
template <typename TType, typename TSizeTraits>
struct dimension<convolution_kernel<TType, TSizeTraits> >
	: dimension_helper<TSizeTraits::size>
{ };


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
	//return get<tDim>(aMask.mTo);
}

template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, int i) 
{
	//return ;
}

template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, int i, int j) 
{
	//return ;
}

template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, int i, int j, int k) 
{
	//return ;
}

/** 
 * @}
 **/

	
namespace detail {

template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_1d_tag) 
{
	TOutputType tmp = 0;
	for (int i = from<0>(aMask); i < to<0>(aMask); ++i) {
		
	}
	return tmp;
}

template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_2d_tag) 
{
	TOutputType tmp = 0;
	for (int j = from<1>(aMask); j < to<1>(aMask); ++j) {
		for (int i = from<0>(aMask); i < to<0>(aMask); ++i) {
			tmp += get(aMask, i, j) * aLocator[i,j];
		}
	}
	return tmp;
}

template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_3d_tag) 
{
	TOutputType tmp = 0;
	for (int k = from<2>(aMask); k < to<2>(aMask); ++k) {
		for (int j = from<1>(aMask); j < to<1>(aMask); ++j) {
			for (int i = from<0>(aMask); i < to<0>(aMask); ++i) {
				tmp += get(aMask, i, j, k) * aLocator[i,j];
			}
		}
	}
	return tmp;
}

template<typename TInputType, typename TOutputType, typename TConvolutionMask>
struct convolution_operator
{
	convolution_operator(const TConvolutionMask&aMask): mask(aMask)
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID TOutputType
	operator()(TLocator aLocator) const
	{
		//TOutputType tmp = 0;
		//tmp.data[0] = abs(aLocator[typename TLocator::diff_t(-1,0)].data[0] - aLocator[typename TLocator::diff_t()].data[0]) + abs(aLocator[typename TLocator::diff_t(0,-1)].data[0] - aLocator[typename TLocator::diff_t()].data[0]);
		//return tmp;
		return apply_convolution(mask, aLocator, typename dimension<TConvolutionMask>::type());
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

CUGIP_FORCE_INLINE convolution_kernel<int, size_traits_2d<3,3> >
laplacian_kernel()
{
	convolution_kernel<int, size_traits_2d<3,3> > tmp;

	return tmp;
}

}//namespace cugip

