#pragma once

#include <cugip/detail/include.hpp>
#include <boost/mpl/bool.hpp>
#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/memory.hpp>


namespace cugip {

struct neighborhood_4
{
	typedef typename dim_traits<2>::diff_t diff_t;
	static const size_t size = 4;

	CUGIP_DECL_HYBRID diff_t
	neighbor_offset(size_t aIdx)
	{
		switch (aIdx) {
		case 0: return diff_t( 0, -1);
		case 1: return diff_t( 1,  0);
		case 2: return diff_t( 0,  1);
		case 3: return diff_t(-1,  0);
		default: CUGIP_ASSERT(false);
		};
		return diff_t();
	}
};

struct neighborhood_4_tag
{
	static const size_t dimension = 2;
	typedef neighborhood_4 type;
};

template<typename TImageLocator, typename TNeighborhood>
class neighborhood_accessor: public TNeighborhood
{
public:
	typedef typename TImageLocator::extents_t extents_t;
	typedef typename TImageLocator::coord_t coord_t;
	typedef typename TImageLocator::diff_t diff_t;
	typedef typename TImageLocator::value_type value_type;
	typedef typename TImageLocator::const_value_type const_value_type;

	CUGIP_DECL_HYBRID
	neighborhood_accessor(TImageLocator aLocator): mLocator(aLocator)
	{ /*empty*/ }

	CUGIP_DECL_HYBRID const_value_type &
	operator[](size_t aIdx)
	{
		CUGIP_ASSERT(aIdx < TNeighborhood::size);
		return mLocator[this->neighbor_offset(aIdx)];
	}
	
protected:
	TImageLocator mLocator;
};


template<typename TImageView, typename TNeigborhoodTag>
struct get_neighborhood_accessor
{
	typedef image_locator<TImageView, border_handling_repeat_t> locator_t;
	typedef neighborhood_accessor<locator_t, typename TNeigborhoodTag::type> type;

};

}//namespace cugip

