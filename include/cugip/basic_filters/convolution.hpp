#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>

namespace cugip {

template <typename TType, typename TSizeTraits>
struct convolution_kernel
{
	TType data[TSizeTraits::size];
};

/** \ingroup  traits
 * @{
 **/
template <typename TType, typename TSizeTraits>
struct dimension<convolution_kernel<TType, TSizeTraits> >
	: cugip::dimension<TSizeTraits>
{ };


template<size_t tDim, typename TConvolutionMask>
struct size_traits;

template<typename TType, typename TSizeTraits>
struct size_traits<0, convolution_kernel<TType, TSizeTraits> >
{
	static const size_t value = TSizeTraits::width;
};

template<typename TType, typename TSizeTraits>
struct size_traits<1, convolution_kernel<TType, TSizeTraits> >
{
	static const size_t value = TSizeTraits::height;
};

template<typename TType, typename TSizeTraits>
struct size_traits<2, convolution_kernel<TType, TSizeTraits> >
{
	static const size_t value = TSizeTraits::depth;
};
/** 
 * @}
 **/


/** \ingroup auxiliary_function
 * @{
 **/


template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE CUGIP_DECL_HYBRID const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, size_t i) 
{
	return aMask.data[i];
}

template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE CUGIP_DECL_HYBRID const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, size_t i, size_t j) 
{
	return aMask.data[i + (j * TSizeTraits::width)];
}

template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE CUGIP_DECL_HYBRID const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, size_t i, size_t j, size_t k) 
{
	return aMask.data[i + (j * TSizeTraits::width) + (k * TSizeTraits::width * TSizeTraits::height)];
}

/** 
 * @}
 **/

	
namespace detail {

template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_1d_tag) 
{
	TOutputType tmp(0);
	for (size_t i = 0; i < size_traits<0, TConvolutionMask>::value; ++i) {
		tmp += get(aMask, i) * aLocator[typename TLocator::diff_t(i)];		
	}
	return tmp;
}

template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_2d_tag) 
{
	TOutputType tmp(0);
	for (size_t j = 0; j < size_traits<1, TConvolutionMask>::value; ++j) {
		for (size_t i = 0; i < size_traits<0, TConvolutionMask>::value; ++i) {
			tmp += get(aMask, i, j) * aLocator[typename TLocator::diff_t(i, j)];
		}
	}
	return tmp;
}

template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_3d_tag) 
{
	TOutputType tmp(0);
	for (size_t k = 0; k < size_traits<2, TConvolutionMask>::value; ++k) {
		for (size_t j = 0; j < size_traits<1, TConvolutionMask>::value; ++j) {
			for (size_t i = 0; i < size_traits<0, TConvolutionMask>::value; ++i) {
				tmp += get(aMask, i, j, k) * aLocator[typename TLocator::diff_t(i, j, k)];
			}
		}
	}
	return tmp;
}

//TODO - addition and multiplication
template<typename TInputType, typename TOutputType, typename TConvolutionMask>
struct convolution_operator
{
	convolution_operator(const TConvolutionMask&aMask, TOutputType aAddition = TOutputType(0)): mask(aMask), addition(aAddition)
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID TOutputType
	operator()(TLocator aLocator) const
	{
		//TOutputType tmp = 0;
		//tmp.data[0] = abs(aLocator[typename TLocator::diff_t(-1,0)].data[0] - aLocator[typename TLocator::diff_t()].data[0]) + abs(aLocator[typename TLocator::diff_t(0,-1)].data[0] - aLocator[typename TLocator::diff_t()].data[0]);
		//return tmp;
		return addition + apply_convolution<TOutputType>(mask, aLocator, typename dimension<TConvolutionMask>::type());
	}

	TConvolutionMask mask;
	TOutputType addition;
};

}//namespace detail


template <typename TInView, typename TOutView, typename TConvolutionMask>
void 
convolution(TInView aInView, TOutView aOutView, TConvolutionMask aConvolutionMask, typename TOutView::value_type aAddition = typename TOutView::value_type(0))
{
	typedef detail::convolution_operator<
		typename TInView::value_type, 
		typename TOutView::value_type, 
		TConvolutionMask> convolution_functor;

	cugip::filter(
			aInView, 
			aOutView, 
			convolution_functor(aConvolutionMask, aAddition)
			);
}

CUGIP_FORCE_INLINE convolution_kernel<int, size_traits_2d<3,3> >
laplacian_kernel()
{
	convolution_kernel<int, size_traits_2d<3,3> > tmp = {0, 1, 0, 1, -4, 1, 0, 1, 0};

	return tmp;
}

}//namespace cugip

