#pragma once
//TODO - delate this file when not needed

#include <cugip/detail/include.hpp>
#include <cugip/math.hpp>
#include <cugip/utils.hpp>
#include <cugip/exception.hpp>
#include <cugip/traits.hpp>

namespace cugip {

template<typename TType, typename TSizeTraits>
struct static_memory_block
{
	static const int size = TSizeTraits::size;
	static const int dimension = TSizeTraits::dimension;
	typedef TSizeTraits intraits;
	typedef typename dim_traits<dimension>::extents_t extents_t;
	typedef typename dim_traits<dimension>::coord_t coord_t;
	typedef TType value_type;

	CUGIP_DECL_HYBRID
	static_memory_block()
	{}

	CUGIP_DECL_HYBRID const value_type &
	operator[](coord_t aCoords) const
	{
		/*int index = TSizeTraits::get_index(aCoords);
		if (index >= size) {
			printf("XXX %d %d %d\n", aCoords[0], aCoords[1], aCoords[2]);
			return mData[0];
		}*/
		return mData[TSizeTraits::get_index(aCoords)];
	}

	CUGIP_DECL_HYBRID value_type &
	operator[](coord_t aCoords)
	{
		/*int index = TSizeTraits::get_index(aCoords);
		if (index >= size) {
			printf("XXX %d %d %d\n", aCoords[0], aCoords[1], aCoords[2]);
			return mData[0];
		}*/
		return mData[TSizeTraits::get_index(aCoords)];
	}


	CUGIP_DECL_HYBRID static extents_t
	dimensions()
	{ return intraits::template get_extents<extents_t>(); }

	value_type mData[TSizeTraits::size];
};



}//namespace cugip
