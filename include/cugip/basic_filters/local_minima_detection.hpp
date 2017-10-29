#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/access_utils.hpp>
#include <cugip/basic_filters/connected_component_labeling.hpp>


namespace cugip {

template<typename TLabel>
struct local_minima_detection_ftor
{
	template <typename TLocator>
	CUGIP_DECL_DEVICE TLabel
	operator()(TLocator aLocator) const
	{
		typename TLocator::value_type value = aLocator.get();
		bool isMinimum = true;
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				typename TLocator::value_type neighbor = aLocator[typename TLocator::diff_t(i, j)];
				if (neighbor < value) {
					isMinimum = false;
				}
			}
		}
		return isMinimum ? (get_linear_access_index(aLocator.dimensions(), aLocator.coords()) + 1) : 0;
	}
};

template<typename TUnionFind>
struct clear_nonminima_plateaus_ftor
{
	clear_nonminima_plateaus_ftor(TUnionFind aUnionFind)
		: mUnionFind(aUnionFind)
	{}

	template <typename TImageLocator, typename TLabelLocator>
	CUGIP_DECL_DEVICE void
	operator()(TImageLocator aImageLocator, TLabelLocator aLabelLocator)
	{
		typename TImageLocator::value_type value = aImageLocator.get();
		typename TLabelLocator::value_type label = aLabelLocator.get();
		if (label != 0) {
			for (int j = -1; j <= 1; ++j) {
				for (int i = -1; i <= 1; ++i) {
					typename TImageLocator::value_type neighbor = aImageLocator[typename TImageLocator::diff_t(i, j)];
					if (neighbor <= value) {
						if (aLabelLocator[typename TLabelLocator::diff_t(i, j)] == 0) {
							mUnionFind.set(label, 0);
						}
					}
				}
			}
		}
	}

	TUnionFind mUnionFind;
};


template <typename TImageView, typename TIdImageView, typename TUnionFind>
void
clear_nonminima_plateaus(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TUnionFind aUnionFind
		)
{
	for_each_locator(aImageView, aIdImageView, clear_nonminima_plateaus_ftor<TUnionFind>(aUnionFind));
	detail::update_labels(aIdImageView, aUnionFind);
}


template <typename TImageView, typename TIdImageView, typename TUnionFind>
void
local_minima_detection(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TUnionFind aUnionFind
		)
{
	filter(aImageView, aIdImageView, local_minima_detection_ftor<typename TIdImageView::value_type>());
	connected_component_labeling(aIdImageView, aUnionFind);
	clear_nonminima_plateaus(aImageView, aIdImageView, aUnionFind);
}



}//namespace cugip

