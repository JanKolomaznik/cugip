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

} // namespace detail


/** \ingroup meta_algorithm
 * @{
 **/

template <int tDimension, typename TFunctor>
TFunctor
for_each(region<tDimension> aRegion, TFunctor aOperator)
{
	return cugip::detail::for_each_implementation(aRegion, aOperator);
}

/**
 * @}
 **/

}  // namespace cugip
