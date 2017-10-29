#pragma once

#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
//#include <cugip/filter.hpp>
#include <cugip/basic_filters/convolution.hpp>

namespace cugip {
namespace detail {

template<typename TOutputType, typename TLocator, int tDim>
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
struct gradient_symmetric_difference
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

template<typename TInputType, typename TOutputType>
struct gradient_magnitude_symmetric_difference
{
	template<typename TLocator>
	CUGIP_DECL_HYBRID TOutputType
	operator()(TLocator aLocator) const
	{
		typedef simple_vector<TInputType, dimension<TLocator>::value> gradient_t;
		gradient_t tmp;
		detail::compute_symmetric_difference<gradient_t, TLocator, dimension<TLocator>::value >::compute(tmp, aLocator);

		return magnitude(tmp);
	}
};

//----------------------------------------------------------------------------------------

namespace detail {

template<typename TOutputType, typename TLocator, int tDim>
struct compute_divergence
{
	static CUGIP_DECL_HYBRID void
	compute(TOutputType &aOutput, TLocator &aLocator)
	{
		aOutput += (get<tDim-1>(aLocator.dim_offset<tDim-1>(0)) - get<tDim-1>(aLocator.dim_offset<tDim-1>(-1)));
		compute_divergence<TOutputType, TLocator, tDim - 1>::compute(aOutput, aLocator);
	}
};

//Stop the compile time recursion
template<typename TOutputType, typename TLocator>
struct compute_divergence<TOutputType, TLocator, 0>
{
	static CUGIP_DECL_HYBRID void
	compute(TOutputType &, TLocator &)
	{ }
};

} //namespace detail


template<typename TInputType, typename TOutputType>
struct divergence
{
	template<typename TLocator>
	CUGIP_DECL_HYBRID TOutputType
	operator()(TLocator aLocator) const
	{
		TOutputType tmp = 0;
		detail::compute_divergence<TOutputType, TLocator, dimension<TLocator>::value >::compute(tmp, aLocator);

		return tmp;
	}
};

template<int tDimension>
struct sobel_gradient_magnitude
{
	sobel_gradient_magnitude()
		: kernel(sobel_gradient_kernel<tDimension>())
	{
		for (int k = -1; k < 2; ++k) {
			for (int j = -1; j < 2; ++j) {
				for (int i = -1; i < 2; ++i) {
					std::cout << kernel.get(vect3i_t(i, j, k)) << "; ";
				}
				std::cout << "\n";
			}
			std::cout << "---------------------------------------------------\n";
		}

	}

	template<typename TLocator>
	CUGIP_DECL_HYBRID float
	operator()(TLocator aLocator) const
	{
		simple_vector<float, tDimension> result;
		apply_convolution_kernel(aLocator, kernel, result);
		return magnitude(result);
	}
	convolution_kernel<simple_vector<float, tDimension>, typename FillStaticSize<tDimension, 3>::Type> kernel;
};

template<int tDimension>
struct sobel_gradient
{
	sobel_gradient()
		: kernel(sobel_gradient_kernel<tDimension>())
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID simple_vector<float, tDimension>
	operator()(TLocator aLocator) const
	{
		simple_vector<float, tDimension> result;
		apply_convolution_kernel(aLocator, kernel, result);
		return result;
	}
	convolution_kernel<simple_vector<float, tDimension>, typename FillStaticSize<tDimension, 3>::Type> kernel;
};

template<int tDimension>
struct sobel_weighted_divergence
{
	typedef convolution_kernel<simple_vector<float, tDimension>, typename FillStaticSize<tDimension, 3>::Type> Kernel;
	sobel_weighted_divergence(float aWeight)
		: weight(aWeight)
		, kernel(sobel_gradient_kernel<tDimension>())
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID float
	operator()(TLocator aLocator) const
	{
		float result = 0.0f;
		typedef simple_vector<int, tDimension> Index;
		for_each_neighbor(
			-kernel.offset,
			kernel.size() - kernel.offset,
			[&](const Index &aIndex) {
				result += dot(aLocator[aIndex], kernel.get(aIndex));
			});
		return weight * result;
	}

	float weight;
	Kernel kernel;
};

}//namespace cugip
