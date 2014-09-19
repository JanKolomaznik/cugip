#pragma once

#include <cugip/image_view.hpp>
#include <cugip/utils.hpp>

namespace cugip {

namespace detail {


} //namespace detail

template<typename TImageView>
typename TImageView::accessed_type
linear_access(TImageView &aView, size_t aIdx)
{
	typename TImageView::coord_t coords;
	for(size_t i = 0; i < dimension<TImageView>::value; ++i) {
	       	coords[i] = aIdx % aView.dimensions()[i];
		aIdx /= aView.dimensions()[i];
	}

	return aView[coords];
}

} //namespace cugip
