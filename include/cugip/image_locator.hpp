#pragma once

#include <cugip/detail/include.hpp>
#include <boost/mpl/bool.hpp>
#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/memory.hpp>


namespace cugip {

enum class border_handling_enum {
	NONE,
	MIRROR,
	REPEAT,
	PERIODIC,
	ZERO
};

/*struct border_handling_none_t
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
};*/

template <border_handling_enum tBorderHandling>
struct BorderHandlingTraits {
	static constexpr border_handling_enum kValue = tBorderHandling;

	CUGIP_HD_WARNING_DISABLE
	template<typename TView>
	CUGIP_DECL_HYBRID
	static typename TView::accessed_type access(
				TView &view,
				const typename TView::coord_t &coordinates,
				const simple_vector<int, dimension<TView>::value> &offset)
	{
		return view[coordinates + offset];
	}
};

template <>
struct BorderHandlingTraits<border_handling_enum::MIRROR> {
	static constexpr border_handling_enum kValue = border_handling_enum::MIRROR;

	CUGIP_HD_WARNING_DISABLE
	template<typename TView>
	CUGIP_DECL_HYBRID
	static typename TView::accessed_type access(
				TView &view,
				const typename TView::coord_t &coordinates,
				const simple_vector<int, dimension<TView>::value> &offset)
	{
		typedef typename TView::IndexType IndexType;
		auto region = ValidRegion(view);
		auto minimum = region.corner; //IndexType();
		auto maximum = minimum + region.size - IndexType::Fill(1);
		auto coords_in_view = coordinates + offset;
		for (int i = 0; i < dimension<TView>::value; ++i) {
			if (coords_in_view[i] < minimum[i]) {
				coords_in_view[i] = minimum[i] + (minimum[i] - coords_in_view[i]);
			} else {
				if (coords_in_view[i] > maximum[i]) {
					coords_in_view[i] = maximum[i] - (coords_in_view[i] - maximum[i]);
				}
			}
		}
		return view[coords_in_view];
	}
};

template <>
struct BorderHandlingTraits<border_handling_enum::REPEAT> {
	static constexpr border_handling_enum kValue = border_handling_enum::REPEAT;

	CUGIP_HD_WARNING_DISABLE
	template<typename TView>
	CUGIP_DECL_HYBRID
	static typename TView::accessed_type access(
				TView &view,
				const typename TView::coord_t &coordinates,
				const simple_vector<int, dimension<TView>::value> &offset)
	{
		typedef typename TView::coord_t coord_t;
		auto region = valid_region(view);
		auto minimum = region.corner; //IndexType();
		auto maximum = region.size - coord_t(1, FillFlag());
		auto coords = min_per_element(maximum, max_per_element(minimum, coordinates + offset));
		return view[coords];;
	}
};

template <>
struct BorderHandlingTraits<border_handling_enum::ZERO> {
	static constexpr border_handling_enum kValue = border_handling_enum::ZERO;

	CUGIP_HD_WARNING_DISABLE
	template<typename TView>
	CUGIP_DECL_HYBRID
	static typename TView::accessed_type access(
				TView &view,
				const typename TView::coord_t &coordinates,
				const simple_vector<int, dimension<TView>::value> &offset)
	{
		typedef typename TView::coord_t coord_t;
		auto region = valid_region(view);
		auto minimum = region.corner; //IndexType();
		auto maximum = region.size - coord_t(1, FillFlag());
		auto index = coordinates + offset;
		for (int i = 0; i < dimension<TView>::value; ++i) {
			if (index[i] < minimum[i] || index[i] > maximum[i]) {
				return 0;
			}
		}
		return view[coordinates + offset];
	}
};


template<typename TImageView, typename TBorderHandling = BorderHandlingTraits<border_handling_enum::NONE>>
class image_locator
{
public:
	static const int cDimension = dimension<TImageView>::value;
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
		return TBorderHandling::access(mView, mCoords, aOffset);
		//coord_t coords = min_coords(mView.dimensions()-coord_t::fill(1), max_coords(coord_t(), mCoords + aOffset));
		//return mView[coords];
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
		coord_t coords;
		cugip::get<tDimIdx/*, typename coord_t::coord_t, coord_t::dim*/>(coords) = aOffset;
		return TBorderHandling::access(mView, mCoords, coords);
		//coords = min_coords(mView.dimensions()-coord_t::fill(1), max_coords(coord_t(), coords));
		//return mView[coords];
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

	CUGIP_DECL_HYBRID 
	TImageView view() const {
		return mView;
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
