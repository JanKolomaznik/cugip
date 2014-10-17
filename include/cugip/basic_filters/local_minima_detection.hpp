#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/access_utils.hpp>
#include <cugip/basic_filters/connected_component_labeling.hpp>


namespace cugip {

struct local_minima_detection_ftor
{
	template <typename TLocator>
	CUGIP_DECL_DEVICE void
	operator()(TLocator aLocator) const
	{
		typename TLocator::value_type value = aLocator.get();
		bool isMinimum = true;
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				typename TLocator::value_type neigbor = aLocator[typename TLocator::diff_t(i, j)];
				if (neighbor < currentMinimum) {
					isMinimum = false;
				}
			}
		}
		return isMinimum ? (get_linear_access_index(aLocator.dimensions(), aLocator.coords()) + 1) : 0;
	}
};

template <typename TImageView, typename TIdImageView, typename TUnionFind>
void
clear_nonminima_plateaus(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TUnionFind aUnionFind
		)
{

}


template <typename TImageView, typename TIdImageView, typename TUnionFind>
void
local_minima_detection(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TUnionFind aUnionFind
		)
{
	filter(aImageView, aIdImageView, local_minima_detection_ftor());
	connected_component_labeling(aIdImageView, aUnionFind);
	clear_nonminima_plateaus(aImageView, aIdImageView, aUnionFind)
}



}//namespace cugip

