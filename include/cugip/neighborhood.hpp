#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>

namespace cugip {

template<int tDim>
struct full_neighborhood;

template<>
struct full_neighborhood<2>
{
	static const int dimension = 2;
	typedef simple_vector<int, dimension> offset_t;
	static const int count = 8;

	CUGIP_DECL_HYBRID static offset_t
	get(int aIndex)
	{
		offset_t cNeighbors[count] = {
			offset_t( -1, -1 ),
			offset_t(  0, -1 ),
			offset_t(  1, -1 ),
			offset_t( -1,  0 ),
			offset_t(  1,  0 ),
			offset_t( -1,  1 ),
			offset_t(  0,  1 ),
			offset_t(  1,  1 )
			};
		return cNeighbors[aIndex];
	}
};


template<>
struct full_neighborhood<3>
{
	static const int dimension = 3;
	typedef simple_vector<int, dimension> offset_t;
	static const int count = 26;

	CUGIP_DECL_HYBRID static offset_t
	get(int aIndex)
	{
		offset_t cNeighbors[count] = {
			offset_t( -1, -1, -1 ),
			offset_t(  0, -1, -1 ),
			offset_t(  1, -1, -1 ),
			offset_t( -1,  0, -1 ),
			offset_t(  0,  0, -1 ),
			offset_t(  1,  0, -1 ),
			offset_t( -1,  1, -1 ),
			offset_t(  0,  1, -1 ),
			offset_t(  1,  1, -1 ),

			offset_t( -1, -1, 0 ),
			offset_t(  0, -1, 0 ),
			offset_t(  1, -1, 0 ),
			offset_t( -1,  0, 0 ),
			offset_t(  1,  0, 0 ),
			offset_t( -1,  1, 0 ),
			offset_t(  0,  1, 0 ),
			offset_t(  1,  1, 0 ),

			offset_t( -1, -1, 1 ),
			offset_t(  0, -1, 1 ),
			offset_t(  1, -1, 1 ),
			offset_t( -1,  0, 1 ),
			offset_t(  0,  0, 1 ),
			offset_t(  1,  0, 1 ),
			offset_t( -1,  1, 1 ),
			offset_t(  0,  1, 1 ),
			offset_t(  1,  1, 1 )
			};
		return cNeighbors[aIndex];
	}
};


}//namespace cugip
