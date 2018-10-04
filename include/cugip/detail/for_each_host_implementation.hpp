#pragma once

#include <cugip/region.hpp>

namespace cugip {
namespace detail {


template <typename TFunctor>
TFunctor
for_each_implementation(region<2> aRegion, TFunctor aOperator)
{
	for (int j = aRegion.corner[1]; j < aRegion.size[1]; ++j) {
		for (int i = aRegion.corner[0]; i < aRegion.size[0]; ++i) {
			aOperator(Int2(i, j));
		}
	}
	return aOperator;
}

template <typename TFunctor>
TFunctor
for_each_implementation(region<3> aRegion, TFunctor aOperator)
{
	for (int k = aRegion.corner[2]; k < aRegion.size[2]; ++k) {
		for (int j = aRegion.corner[1]; j < aRegion.size[1]; ++j) {
			for (int i = aRegion.corner[0]; i < aRegion.size[0]; ++i) {
				aOperator(Int3(i, j, k));
			}
		}
	}
	return aOperator;
}


template <typename TInView, typename TFunctor, typename TPolicy>
void
for_each_host(TInView aInView, TFunctor aOperator, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		aOperator(linear_access(aInView, i));
	}
}

template<>
struct ForEachImplementation<false> {
	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::for_each_host(aInView, aOperator, aPolicy);
	}
};


template <typename TInView, typename TFunctor, typename TPolicy>
void
for_each_position_host(TInView aInView, TFunctor aOperator, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		auto index = index_from_linear_access_index(aInView, i);
		aOperator(aInView[index], index);
	}
}

template<>
struct ForEachPositionImplementation<false> {
	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::for_each_position_host(aInView, aOperator, aPolicy);
	}

	template <int tDimension, typename TFunctor/*, typename TPolicy*/>
	static void run(region<tDimension> aRegion, TFunctor aOperator/*, TPolicy aPolicy, cudaStream_t aCudaStream*/) {
		for_each_implementation(aRegion, aOperator);
	}

};




}//namespace detail
}//namespace cugip
