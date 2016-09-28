#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/static_int_sequence.hpp>
#include <cugip/neighborhood_accessor.hpp>

namespace cugip {

template <typename TType, typename TSizeTraits>
struct convolution_kernel
{
	static constexpr int cDimension = TSizeTraits::cDimension;
	typedef TType element_t;

	CUGIP_DECL_HYBRID
	static constexpr simple_vector<int, cDimension> size()
	{
		return to_vector(TSizeTraits());
	}

	CUGIP_DECL_HYBRID
	TType get(const simple_vector<int, cDimension> &aIndex) const
	{
		return data[get_linear_access_index(to_vector(TSizeTraits()), aIndex + offset)];
	}

	CUGIP_DECL_HYBRID
	TType & get(const simple_vector<int, cDimension> &aIndex)
	{
		return data[get_linear_access_index(to_vector(TSizeTraits()), aIndex + offset)];
	}

	TType data[TSizeTraits::count()];
	simple_vector<int, cDimension> offset;
};

template <typename TType, int tSize>
struct convolution_kernel<TType, StaticSize<tSize>>
{
	static constexpr int cDimension = 1;
	typedef TType element_t;

	CUGIP_DECL_HYBRID
	static constexpr int size()
	{
		return tSize;
	}

	CUGIP_DECL_HYBRID
	TType get(int aIndex) const
	{
		return data[aIndex + offset];
	}

	CUGIP_DECL_HYBRID
	TType & get(int aIndex)
	{
		return data[aIndex + offset];
	}

	TType data[tSize];
	int offset;
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
		to_vector(typename FillStaticSize<tDimension, 3>::Type()),
	//for_each_in_radius<>
		[&](Index &aIndex){
			simple_vector<float, tDimension> value;
			for (int i = 0; i < tDimension; ++i) {
				value[i] = derivative[aIndex[i]];
				std::cout << value << "; ";
				for (int j = 0; j < tDimension; ++j) {
					if (i != j) {
						value[i] *= smooth[aIndex[j]];
						//std::cout << value << "; ";
					}
				}
			}
			//std::cout << value << "; " << aIndex << '\n';
			result.get(aIndex) = value;
		});
	result.offset = simple_vector<float, tDimension>(1, FillFlag());
	return result;
}

template<int tRadius>
convolution_kernel<float, StaticSize<2 * tRadius + 1>>
gaussian_kernel()
{
	convolution_kernel<float, StaticSize<2 * tRadius + 1>> kernel;
	kernel.offset = tRadius;

	float sigmaSqr = sqr((2 * tRadius + 1) / 4.0f);
	float sum = 0.0f;
	for (int i = -tRadius; i < tRadius; ++i) {
		float value = exp(-(sqr(i)/(2*sigmaSqr)));
		sum += value;
		kernel.get(i) = value;
	}
	auto factor = 1.0f / sum;
	for (int i = -tRadius; i < tRadius; ++i) {
		kernel.get(i) *= factor;
	}

	return kernel;
}

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

template<typename TKernel>
struct convolution_operator
{
	template<typename TLocator>
	CUGIP_DECL_HYBRID auto
	operator()(TLocator aLocator) const
		-> decltype(std::declval<typename TLocator::const_value_type>() * std::declval<typename TKernel::element_t>())
	{
		decltype(std::declval<typename TLocator::const_value_type>() * std::declval<typename TKernel::element_t>()) result;
		apply_convolution_kernel(aLocator, kernel, result);
		return result;
	}
	TKernel kernel;
};

template<typename TKernel, int tDimension>
struct separable_convolution_operator
{
	template<typename TLocator>
	CUGIP_DECL_HYBRID auto
	operator()(TLocator aLocator) const
		-> typename TLocator::const_value_type
	{
		auto result = zero<typename TLocator::const_value_type>();
		typename TLocator::diff_t offset;
		for (offset[tDimension] = -kernel.offset; offset[tDimension] < kernel.size() - kernel.offset; ++offset[tDimension]) {
			result += aLocator[offset] * kernel.get(offset[tDimension]);
		}
		return result;
	}
	TKernel kernel;
};

template<typename TInputView, typename TOutputView, typename TKernel>
void
convolution(TInputView aInView, TOutputView aOutView, TKernel aKernel)
{
	transform_locator(aInView, aOutView, convolution_operator<TKernel>{aKernel});
}

namespace detail {
template<int tDimension>
struct separable_convolution_impl;

template<>
struct separable_convolution_impl<2> {
	template<typename TInputView, typename TOutputView, typename TTmpView, typename TKernel>
	static void
	run(TInputView aInView, TOutputView aOutView, TTmpView aTmpView, TKernel aKernel)
	{
		transform_locator(aInView, aTmpView, separable_convolution_operator<TKernel, 0>{ aKernel });
		transform_locator(aTmpView, aOutView, separable_convolution_operator<TKernel, 1>{ aKernel });
	}
};

template<>
struct separable_convolution_impl<3> {
	template<typename TInputView, typename TOutputView, typename TTmpView, typename TKernel>
	static void
	run(TInputView aInView, TOutputView aOutView, TTmpView aTmpView, TKernel aKernel)
	{
		transform_locator(aInView, aOutView, separable_convolution_operator<TKernel, 0>{ aKernel });
		transform_locator(aOutView, aTmpView, separable_convolution_operator<TKernel, 1>{ aKernel });
		transform_locator(aTmpView, aOutView, separable_convolution_operator<TKernel, 2>{ aKernel });
	}
};

} // namespace detail

template<typename TInputView, typename TOutputView, typename TTmpView, typename TKernel>
void
separable_convolution(TInputView aInView, TOutputView aOutView, TTmpView aTmpView, TKernel aKernel)
{
	detail::separable_convolution_impl<dimension<TInputView>::value>::run(aInView, aOutView, aTmpView, aKernel);
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
