#pragma once

#include <cugip/detail/include.hpp>
#include <boost/mpl/bool.hpp>
#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/memory.hpp>


namespace cugip {

enum border_handling_enum {
	bhNONE,
	bhMIRROR,
	bhREPAT,
	bhPERIODIC
};

struct border_handling_none_t
{
	static const border_handling_enum value = bhNONE;
};

struct border_handling_mirror_t
{
	static const border_handling_enum value = bhMIRROR;
};

struct border_handling_repeat_t
{
	static const border_handling_enum value = bhREPAT;
};

struct border_handling_periodic_t
{
	static const border_handling_enum value = bhPERIODIC;
};


template<typename TImageView, typename TBorderHandling = border_handling_none_t>
class image_locator
{
public:
	typedef typename TImageView::extents_t extents_t;
	typedef typename TImageView::coord_t coord_t;
	typedef typename TImageView::diff_t diff_t;
	typedef typename TImageView::value_type value_type;
	typedef typename TImageView::const_value_type const_value_type;

	typedef typename TImageView::accessed_type accessed_type;

	CUGIP_DECL_HYBRID
	image_locator(TImageView aView, coord_t aCoords): mView(aView), mCoords(aCoords)
	{}

	CUGIP_DECL_HYBRID accessed_type
	operator[](diff_t aOffset)
	{
		//TODO
		coord_t coords = min_coords(mView.dimensions()-coord_t::fill(1), max_coords(coord_t(), mCoords + aOffset));
		return mView[coords];
	}

	CUGIP_DECL_HYBRID accessed_type
	get()
	{
		return mView[mCoords];
	}

	template <int tDimIdx>
	CUGIP_DECL_HYBRID accessed_type
	dim_offset(int aOffset)
	{
		coord_t coords = mCoords;
		cugip::get<tDimIdx/*, typename coord_t::coord_t, coord_t::dim*/>(coords) += aOffset;
		coords = min_coords(mView.dimensions()-coord_t::fill(1), max_coords(coord_t(), coords));
		return mView[coords];
	}

	CUGIP_DECL_HYBRID coord_t
	coords() const
	{
		return mCoords;
	}

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{
		return mView.dimensions();
	}

protected:
	TImageView mView;
	coord_t mCoords;
};

/** \ingroup  traits
 * @{
 **/

template<typename TImageView, typename TBorderHandling>
struct dimension<image_locator<TImageView, TBorderHandling> >: dimension<TImageView> {};


template<typename TImageView, typename TBorderHandling>
CUGIP_DECL_HYBRID image_locator<TImageView, TBorderHandling>
create_locator(TImageView aView, typename TImageView::coord_t aCoords)
{
	return image_locator<TImageView, TBorderHandling>(aView, aCoords);
}

/**
 * @}
 **/


}//namespace cugip
