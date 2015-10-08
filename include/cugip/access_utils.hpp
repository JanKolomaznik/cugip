#pragma once

#include <cugip/math.hpp>
#include <cugip/utils.hpp>

namespace cugip {

namespace detail {


} //namespace detail


/// \return Strides for memory without padding.
CUGIP_DECL_HYBRID
inline Int2 stridesFromSize(Int2 size) {
	return Int2(1, size[0]);
}


/// \return Strides for memory without padding.
CUGIP_DECL_HYBRID
inline Int3 stridesFromSize(Int3 size) {
	return Int3(1, size[0], size[0] * size[1]);
}



CUGIP_HD_WARNING_DISABLE
template<typename TImageView>
CUGIP_DECL_HYBRID auto
linear_access(const TImageView &aView, int aIdx) -> typename TImageView::accessed_type
{
	typename TImageView::coord_t coords;
	for(int i = 0; i < dimension<TImageView>::value; ++i) {
		coords[i] = aIdx % aView.dimensions()[i];
		aIdx /= aView.dimensions()[i];
	}

	return aView[coords];
}

template<typename TExtents, typename TCoordinates>
CUGIP_DECL_HYBRID int
get_linear_access_index(
		TExtents aExtents,
		TCoordinates aCoordinates)
{
	int dim = dimension<TExtents>::value;
	int idx = 0;
	int stride = 1;
	for(int i = 0; i < dim; ++i) {
		idx += aCoordinates[i] * stride;
		stride *= aExtents[i];
	}
	return idx;
}

} //namespace cugip
