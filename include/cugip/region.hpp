#pragma once

#include <cugip/math.hpp>
#include <cugip/access_utils.hpp>

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
	return aRegion.corner + index_from_linear_access_index<tDimension>(aRegion.size, aIdx);
}


}  // namespace cugip
