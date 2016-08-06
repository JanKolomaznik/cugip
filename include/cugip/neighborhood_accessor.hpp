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
		//CUGIP_ASSERT(aIdx < TNeighborhood::size);
		return mLocator[this->neighbor_offset(aIdx)];
	}

protected:
	TImageLocator mLocator;
};


template<typename TImageView, typename TNeigborhoodTag>
struct get_neighborhood_accessor
{
	typedef image_locator<TImageView, BorderHandlingTraits<border_handling_enum::REPEAT>> locator_t;
	typedef neighborhood_accessor<locator_t, typename TNeigborhoodTag::type> type;

};

CUGIP_HD_WARNING_DISABLE
template<typename TCallable>
CUGIP_DECL_HYBRID
void for_each_neighbor(simple_vector<int, 2> aRadius, TCallable aCallable)
{
	simple_vector<int, 2> index;
	for(index[1] = -aRadius[1]; index[1] <= aRadius[1]; ++index[1]) {
		for(index[0] = -aRadius[0]; index[0] <= aRadius[0]; ++index[0]) {
			aCallable(index);
		}
	}
}

CUGIP_HD_WARNING_DISABLE
template<typename TCallable>
CUGIP_DECL_HYBRID
void for_each_neighbor(simple_vector<int, 3> aRadius, TCallable aCallable)
{
	simple_vector<int, 3> index;
	for(index[2] = -aRadius[2]; index[2] <= aRadius[2]; ++index[2]) {
		for(index[1] = -aRadius[1]; index[1] <= aRadius[1]; ++index[1]) {
			for(index[0] = -aRadius[0]; index[0] <= aRadius[0]; ++index[0]) {
				aCallable(index);
			}
		}
	}
}

CUGIP_HD_WARNING_DISABLE
template<typename TCallable>
CUGIP_DECL_HYBRID
void for_each_neighbor(simple_vector<int, 2> aFrom, simple_vector<int, 2> aTo, TCallable aCallable)
{
	simple_vector<int, 2> index;
	for(index[1] = aFrom[1]; index[1] < aTo[1]; ++index[1]) {
		for(index[0] = aFrom[0]; index[0] < aTo[0]; ++index[0]) {
			aCallable(index);
		}
	}
}

CUGIP_HD_WARNING_DISABLE
template<typename TCallable>
CUGIP_DECL_HYBRID
void for_each_neighbor(simple_vector<int, 3> aFrom, simple_vector<int, 3> aTo, TCallable aCallable)
{
	simple_vector<int, 3> index;
	for(index[2] = aFrom[2]; index[2] < aTo[2]; ++index[2]) {
		for(index[1] = aFrom[1]; index[1] < aTo[1]; ++index[1]) {
			for(index[0] = aFrom[0]; index[0] < aTo[0]; ++index[0]) {
				aCallable(index);
			}
		}
	}
}

//TODO - cleanuup the unroll
CUGIP_HD_WARNING_DISABLE
template<int tRadius, typename TCallable>
CUGIP_DECL_DEVICE
void for_each_in_radius(TCallable aCallable)
{
	simple_vector<int, 3> index;
	for(index[2] = -tRadius; index[2] <= tRadius; ++index[2]) {
			#pragma unroll
		for(index[1] = -tRadius; index[1] <= tRadius; ++index[1]) {
			#pragma unroll
			for(index[0] = -tRadius; index[0] <= tRadius; ++index[0]) {
				aCallable(index);
			}
		}
	}
}

CUGIP_HD_WARNING_DISABLE
template<int tRadius, typename TCallable>
CUGIP_DECL_DEVICE
void for_each_in_radius2(TCallable aCallable)
{
	int yOffset = (threadIdx.y + 1) % 2;
	simple_vector<int, 3> index;
	for(index[2] = -tRadius; index[2] <= tRadius; ++index[2]) {
		//for(index[1] = -tRadius; index[1] <= tRadius; ++index[1]) {
		for(int tmp = -tRadius + yOffset; tmp <= tRadius + yOffset; ++tmp) {
			index[1] = (tmp + tRadius) % (2*tRadius + 1) - tRadius;
			for(index[0] = -tRadius; index[0] <= tRadius; ++index[0]) {
				aCallable(index);
			}
		}
	}
}
}//namespace cugip
