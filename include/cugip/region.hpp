#pragma once

#include <cugip/math.hpp>

namespace cugip {

template<int tDimension>
struct region
{
	simple_vector<int, tDimension> corner;
	simple_vector<int, tDimension> size;
};

template<int tDimension>
CUGIP_DECL_HYBRID
bool
isInsideRegion(const simple_vector<int, tDimension> &aSize, const simple_vector<int, tDimension> &aCoords)
{
	return aCoords >= simple_vector<int, tDimension>() && aCoords < aSize;
}

CUGIP_HD_WARNING_DISABLE
template<int tDimension>
CUGIP_DECL_HYBRID auto
region_linear_access(const region<tDimension> &aRegion, int aIdx) -> simple_vector<int, tDimension>
{
	simple_vector<int, tDimension> coords;
	for(int i = 0; i < tDimension; ++i) {
		coords[i] = aIdx % region.size[i];
		aIdx /= region.size[i];
	}

	return region.corner + coords;
}


}  // namespace cugip
