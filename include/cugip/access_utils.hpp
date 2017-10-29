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

/// \return Strides for memory without padding.
CUGIP_DECL_HYBRID
inline int stridesFromSize(int size) {
	return 1;
}

CUGIP_HD_WARNING_DISABLE
template<int tDimension>
CUGIP_DECL_HYBRID simple_vector<int, tDimension>
index_from_linear_access_index(const simple_vector<int, tDimension> &aExtents, int aIdx)
{
	CUGIP_ASSERT(multiply(aExtents) > aIdx);
	CUGIP_ASSERT(aIdx >= 0);
	simple_vector<int, tDimension> coords;
	for(int i = 0; i < tDimension; ++i) {
		coords[i] = aIdx % aExtents[i];
		aIdx /= aExtents[i];
	}
	return coords;
}


CUGIP_HD_WARNING_DISABLE
template<typename TImageView>
CUGIP_DECL_HYBRID auto
index_from_linear_access_index(const TImageView &aView, int aIdx) -> typename TImageView::coord_t
{
	return index_from_linear_access_index(aView.dimensions(), aIdx);
}

CUGIP_HD_WARNING_DISABLE
template<typename TImageView>
CUGIP_DECL_HYBRID auto
linear_access(const TImageView &aView, int aIdx) -> typename TImageView::accessed_type
{
	return aView[index_from_linear_access_index(aView, aIdx)];
}

template<typename TExtents, typename TCoordinates>
CUGIP_DECL_HYBRID
inline int64_t linear_index_from_strides(
		TExtents aStrides,
		TCoordinates aCoordinates)
{
	static_assert(dimension<TExtents>::value == dimension<TCoordinates>::value, "strides and coordinates must be of same dimensionality");
	int64_t linear_index = 0;
	for (int i =  0; i < dimension<TExtents>::value; ++i) {
		linear_index += int64_t(aStrides[i]) * aCoordinates[i];
	}
	return linear_index;
}

template<typename TExtents, typename TCoordinates>
CUGIP_DECL_HYBRID int64_t
get_linear_access_index(
		TExtents aExtents,
		TCoordinates aCoordinates)
{
	int dim = dimension<TExtents>::value;
	int64_t idx = 0;
	int64_t stride = 1;
	for(int i = 0; i < dim; ++i) {
		idx += aCoordinates[i] * stride;
		stride *= aExtents[i];
	}
	return idx;
}

//TODO handle constness, add static assertions, compatible with array view
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


// method to seperate bits from a given integer 3 positions apart
inline uint64_t splitBy3(unsigned int a){
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

template<typename TExtents, typename TCoordinates>
CUGIP_DECL_HYBRID int
get_zorder_access_index(
		TExtents aExtents,
		TCoordinates aCoordinates)
{
	static_assert(dimension<TExtents>::value == 3, "TODO: 2 dimensions");
	uint64_t answer = splitBy3(aCoordinates[0]) | (splitBy3(aCoordinates[1]) << 1) | (splitBy3(aCoordinates[2]) << 2);
	return answer;
}

/*template<typename TExtents, typename TCoordinates>
CUGIP_DECL_HYBRID int
get_blocked_order_access_index(
		TExtents aExtents,
		TCoordinates aCoordinates)
{
	static_assert(dimension<TExtents>::value == 3, "TODO: 2 dimensions");
	TCoordinates corner = 2 * div(aCoordinates, 2);
	int offset = aExtents[0] * aExtents[1] * corner[2];
	int zMultiplier = 2;
	if ((aExtents[2] % 2) && (aCoordinates[2] == (aExtents[2] - 1))) {
		zMultiplier = 1;
	}
	offset += aExtents[0] * corner[1] * zMultiplier;

	int yMultiplier = 2;
	if ((aExtents[1] % 2) && (aCoordinates[1] == (aExtents[1] - 1))) {
		yMultiplier = 1;
	}
	offset += corner[0] * zMultiplier * yMultiplier;

	auto blockSize = min_per_element(Int3(2, 2, 2), aExtents - corner);
	auto blockPos = aCoordinates - corner;
	offset += blockPos[0] + blockPos[1] * blockSize[0] + blockPos[2] * blockSize[0] * blockSize[1];
	return offset;
}*/


template<int tBlockSize, typename TExtents, typename TCoordinates>
CUGIP_DECL_HYBRID int
get_blocked_order_access_index(
		TExtents aExtents,
		TCoordinates aCoordinates)
{
	static_assert(dimension<TExtents>::value == 3, "TODO: 2 dimensions");
	TCoordinates corner = tBlockSize * div(aCoordinates, tBlockSize);
	int offset = aExtents[0] * aExtents[1] * corner[2];
	/*int zMultiplier = 2;
	if ((aExtents[2] % 2) && (aCoordinates[2] == (aExtents[2] - 1))) {
		zMultiplier = 1;
	}
	offset += aExtents[0] * corner[1] * zMultiplier;

	int yMultiplier = 2;
	if ((aExtents[1] % 2) && (aCoordinates[1] == (aExtents[1] - 1))) {
		yMultiplier = 1;
	}
	offset += corner[0] * zMultiplier * yMultiplier;*/

	auto blockSize = min_per_element(Int3(tBlockSize, tBlockSize, tBlockSize), aExtents - corner);
	auto blockPos = aCoordinates - corner;

	offset += aExtents[0] * corner[1] * blockSize[2];
	offset += corner[0] * blockSize[1] * blockSize[2];
	offset += blockPos[0] + blockPos[1] * blockSize[0] + blockPos[2] * blockSize[0] * blockSize[1];
	return offset;
}


} //namespace cugip
