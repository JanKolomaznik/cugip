#pragma once

#include <cugip/math/vector.hpp>
#include <cugip/access_utils.hpp>
#include <iostream>

namespace cugip {

// TODO region constructors to allow template deduction
template<int tDimension>
struct region
{
	simple_vector<int, tDimension> corner;
	simple_vector<int, tDimension> size;
};

template<>
struct region<1>
{
	int64_t corner;
	int64_t size;
};

template<int tDimension>
CUGIP_DECL_HYBRID
bool
isInsideRegion(const simple_vector<int, tDimension> &aSize, const simple_vector<int, tDimension> &aCoords)
{
	return aCoords >= simple_vector<int, tDimension>() && aCoords < aSize;
}

template<int tDimension>
CUGIP_DECL_HYBRID
bool
isInsideRegion(const region<tDimension> &aRegion, const simple_vector<int, tDimension> &aCoords)
{
	return aCoords >= aRegion.corner && aCoords < aRegion.corner + aRegion.size;
}

CUGIP_HD_WARNING_DISABLE
template<int tDimension>
CUGIP_DECL_HYBRID auto
region_linear_access(const region<tDimension> &aRegion, int aIdx) -> simple_vector<int, tDimension>
{
	return aRegion.corner + index_from_linear_access_index<tDimension>(aRegion.size, aIdx);
}

template<int tDimension>
CUGIP_DECL_HYBRID
simple_vector<int, tDimension>
get_corner(const region<tDimension> &aRegion, uint16_t aCornerId) {
	CUGIP_ASSERT(aCornerId < pow_n(2, tDimension));
	// TODO check and unit test
	simple_vector<int, tDimension> result = aRegion.corner;

	for (int i = 0; i < tDimension; ++i) {
		if (aCornerId & (1 << i)) {
			result[i] += aRegion.size[i];
		}
	}
	return result;
}

template<int tDimension>
std::ostream &
operator<<(std::ostream &aStream, const region<tDimension> &aRegion)
{
	return aStream << "{ \"corner\": " << aRegion.corner << ", \"size\": " << aRegion.size << "}";
}

}  // namespace cugip
