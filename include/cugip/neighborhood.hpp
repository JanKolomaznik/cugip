#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>

namespace cugip {

template<size_t tDim>
struct full_neighborhood;

template<>
struct full_neighborhood<2>
{
	static const size_t dimension = 2;
	typedef simple_vector<int, dimension> offset_t;
	static const size_t count = 8;

	CUGIP_DECL_HYBRID static offset_t
	get(size_t aIndex)
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
	static const size_t dimension = 3;
	typedef simple_vector<int, dimension> offset_t;
	static const size_t count = 26;

	CUGIP_DECL_HYBRID static offset_t
	get(size_t aIndex)
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
