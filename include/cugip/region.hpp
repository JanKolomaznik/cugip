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


}  // namespace cugip
