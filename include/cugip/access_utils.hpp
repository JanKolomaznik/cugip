#pragma once

#include <cugip/image_view.hpp>
#include <cugip/utils.hpp>

namespace cugip {

namespace detail {


} //namespace detail

template<typename TImageView>
CUGIP_DECL_HYBRID typename TImageView::accessed_type
linear_access(TImageView &aView, size_t aIdx)
{
	typename TImageView::coord_t coords;
	for(size_t i = 0; i < dimension<TImageView>::value; ++i) {
	       	coords[i] = aIdx % aView.dimensions()[i];
		aIdx /= aView.dimensions()[i];
	}

	return aView[coords];
}

template<typename TExtents, typename TCoordinates>
CUGIP_DECL_HYBRID size_t
get_linear_access_index(
		TExtents aExtents,
		TCoordinates aCoordinates)
{
	int dim = dimension<TExtents>::value;
	size_t idx = 0;
	size_t stride = 1;
	for(size_t i = 0; i < dim; ++i) {
	       	idx += aCoordinates[i] * stride;
		stride *= aExtents[i];
	}
	return idx;
}

} //namespace cugip
