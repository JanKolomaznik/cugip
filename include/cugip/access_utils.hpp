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

//TODO handle constness, add static assertions
template<typename TImageView>
struct device_linear_access_view
{
	CUGIP_DECL_HYBRID
	int
	size() const
	{
		return product(mImageView.dimensions());
	}

	CUGIP_DECL_DEVICE
	typename TImageView::accessed_type
	operator[](int aIndex)
	{
		//TODO prevent repeated computation of size
		return linear_access(mImageView, aIndex);
	}
	TImageView mImageView;
};

template<typename TImageView>
struct host_linear_access_view
{
	//TODO iterator access
	int
	size() const
	{
		return product(mImageView.dimensions());
	}

	typename TImageView::accessed_type
	operator[](int aIndex)
	{
		//TODO prevent repeated computation of size
		return linear_access(mImageView, aIndex);
	}
	TImageView mImageView;
};

//TODO handle device branch
template<typename TImageView>
host_linear_access_view<TImageView>
linear_access_view(TImageView aView)
{
	return { aView };
}


} //namespace cugip
