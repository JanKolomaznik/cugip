#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
//#include <cugip/filter.hpp>
#include <cugip/static_int_sequence.hpp>
#include <cugip/neighborhood_accessor.hpp>

namespace cugip {

template <typename TType, typename TSizeTraits>
struct convolution_kernel
{
	static constexpr int cDimension = TSizeTraits::cDimension;

	CUGIP_DECL_HYBRID
	static constexpr simple_vector<int, cDimension> size()
	{
		return TSizeTraits::vector();
	}

	CUGIP_DECL_HYBRID
	TType get(const simple_vector<int, cDimension> &aIndex) const
	{
		return data[get_linear_access_index(TSizeTraits::vector(), aIndex + offset)];
	}

	CUGIP_DECL_HYBRID
	TType & get(const simple_vector<int, cDimension> &aIndex)
	{
		return data[get_linear_access_index(TSizeTraits::vector(), aIndex + offset)];
	}

	TType data[TSizeTraits::count()];
	simple_vector<int, cDimension> offset;
};

template<int tDimension>
convolution_kernel<simple_vector<float, tDimension>, typename FillStaticSize<tDimension, 3>::Type>
sobel_gradient_kernel()
{
	typedef simple_vector<int, tDimension> Index;
	static constexpr std::array<float, 3> smooth = { 1.0f, 2.0f, 1.0f };
	static constexpr std::array<float, 3> derivative = { -1.0f, 0.0f, 1.0f };
	convolution_kernel<simple_vector<float, tDimension>, typename FillStaticSize<tDimension, 3>::Type> result;

	for_each_neighbor(
		Index(),
		FillStaticSize<tDimension, 3>::Type::vector(),
		[&](Index &aIndex){
			simple_vector<float, tDimension> value;
			for (int i = 0; i < tDimension; ++i) {
				value[i] = derivative[i];
				for (int j = 0; j < tDimension; ++j) {
					if (i != j) {
						value[i] *= smooth[j];
					}
				}
			}
			result.get(aIndex) = value;
		});
	return result;
}
struct A
{
	template<typename T>
	CUGIP_DECL_HYBRID
	void operator()(T){}
};

template<typename TLocator, typename TKernel, typename TResult>
CUGIP_DECL_HYBRID
void apply_convolution_kernel(TLocator aLocator, TKernel aKernel, TResult &aResult)
{
	typedef simple_vector<int, TKernel::cDimension> Index;
	for_each_neighbor(
		-aKernel.offset,
		aKernel.size() - aKernel.offset,
		[&](const Index &aIndex) {
			aResult += aLocator[aIndex] * aKernel.get(aIndex);
		});
}

/*template <typename TType, typename TSizeTraits>
CUGIP_DECL_HOST inline std::ostream &
operator<<( std::ostream &stream, const convolution_kernel<TType, TSizeTraits> &kernel )
{
	//TODO - for all dimension
	for (int i = 0; i < TSizeTraits::height; ++i) {
		stream << "|\t";

		for (int j = 0; j < TSizeTraits::width; ++j) {
			stream << kernel.data[i*TSizeTraits::width + j] << "\t";
		}
		stream << "|\n";
	}
	return stream;
}*/


/** \ingroup  traits
 * @{
 **/
/*template <typename TType, typename TSizeTraits>
struct dimension<convolution_kernel<TType, TSizeTraits> >
	: cugip::dimension<TSizeTraits>
{ };


template<int tDim, typename TConvolutionMask>
struct intraits;

template<typename TType, typename TSizeTraits>
struct intraits<0, convolution_kernel<TType, TSizeTraits> >
{
	static const int value = TSizeTraits::width;
};

template<typename TType, typename TSizeTraits>
struct intraits<1, convolution_kernel<TType, TSizeTraits> >
{
	static const int value = TSizeTraits::height;
};

template<typename TType, typename TSizeTraits>
struct intraits<2, convolution_kernel<TType, TSizeTraits> >
{
	static const int value = TSizeTraits::depth;
};*/
/**
 * @}
 **/


/** \ingroup auxiliary_function
 * @{
 **/


/*template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE CUGIP_DECL_HYBRID const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, int i)
{
	return aMask.data[i];
}

template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE CUGIP_DECL_HYBRID const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, int i, int j)
{
	return aMask.data[i + (j * TSizeTraits::width)];
}

template <typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE CUGIP_DECL_HYBRID const TType &
get(const convolution_kernel<TType, TSizeTraits>& aMask, int i, int j, int k)
{
	return aMask.data[i + (j * TSizeTraits::width) + (k * TSizeTraits::width * TSizeTraits::height)];
}*/

/**
 * @}
 **/


namespace detail {

/*template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_1d_tag)
{
	TOutputType tmp(0);
	for (int i = 0; i < intraits<0, TConvolutionMask>::value; ++i) {
		tmp += get(aMask, i) * aLocator[typename TLocator::diff_t(i)];
	}
	return tmp;
}

template<typename TOutputType, typename TConvolutionMask, typename TLocator>
CUGIP_DECL_HYBRID TOutputType
apply_convolution(const TConvolutionMask &aMask, TLocator &aLocator, dimension_2d_tag)
{
	TOutputType tmp(0);
	for (int j = 0; j < intraits<1, TConvolutionMask>::value; ++j) {
		for (int i = 0; i < intraits<0, TConvolutionMask>::value; ++i) {
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
	for (int k = 0; k < intraits<2, TConvolutionMask>::value; ++k) {
		for (int j = 0; j < intraits<1, TConvolutionMask>::value; ++j) {
			for (int i = 0; i < intraits<0, TConvolutionMask>::value; ++i) {
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
};*/

}//namespace detail


/*template <typename TInView, typename TOutView, typename TConvolutionMask>
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

CUGIP_FORCE_INLINE convolution_kernel<int, intraits_2d<3,3> >
laplacian_kernel()
{
	convolution_kernel<int, intraits_2d<3,3> > tmp = {0, 1, 0, 1, -4, 1, 0, 1, 0};

	return tmp;
}

template<typename TType, typename TSizeTraits>
CUGIP_FORCE_INLINE convolution_kernel<TType, TSizeTraits >
gaussian_kernel()
{
	convolution_kernel<TType, TSizeTraits > tmp;

	float sigma1sqr = sqr(TSizeTraits::width / 4.0f);
	float sigma2sqr = sqr(TSizeTraits::height / 4.0f);

	int centerX = TSizeTraits::width / 2;
	int centerY = TSizeTraits::height / 2;

	TType sum = 0.0f;
	for (int i = 0; i < TSizeTraits::height; ++i) {
		for (int j = 0; j < TSizeTraits::width; ++j) {
			TType value = exp(-(sqr(i-centerX)/(2*sigma1sqr) + sqr(j-centerY)/(2*sigma2sqr)));
			sum += value;
			tmp.data[i*TSizeTraits::width + j] = value;
		}
	}
	TType factor = 1.0f / sum;
	for (int i = 0; i < TSizeTraits::size; ++i) {
		tmp.data[i] *= factor;
	}

	return tmp;
}*/

}//namespace cugip
