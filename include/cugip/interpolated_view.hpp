#pragma once

#include <cugip/image_locator.hpp>

namespace cugip {

template<typename TValue, typename TWeight>
CUGIP_DECL_HYBRID
TValue lerp(TValue value1, TValue value2, TWeight weight) {
	return (1.0f - weight) * value1 + weight * value2;
}

namespace detail {
template<typename TBoundaryHandler, typename TView>
struct Accessor {
	using value_type = typename TView::value_type;

	template<typename TIndex>
	CUGIP_DECL_HYBRID
	value_type operator[](TIndex aIndex) {
		return TBoundaryHandler::access(view, aIndex, TIndex());
	}
	TView view;
};

template<typename TBoundaryHandler, typename TView>
CUGIP_DECL_HYBRID
Accessor<TBoundaryHandler, TView>
make_accessor(TView view) {
	return Accessor<TBoundaryHandler, TView>{ view };
}


template<int tDimension>
struct LinearInterpolationImpl
{
	template<typename TAccessor, typename TWeight, typename TIndex>
	CUGIP_DECL_HYBRID
	static typename TAccessor::value_type compute(TAccessor accessor, TWeight weight, TIndex corner1, TIndex corner2) {
		auto tmp_corner2 = corner2;
		tmp_corner2[tDimension - 1] = corner1[tDimension - 1];

		auto tmp_corner1 = corner1;
		tmp_corner1[tDimension - 1] = corner2[tDimension - 1];
		return lerp(
			LinearInterpolationImpl<tDimension - 1>::compute(accessor, weight, corner1, tmp_corner2),
			LinearInterpolationImpl<tDimension - 1>::compute(accessor, weight, tmp_corner1, corner2),
			weight[tDimension - 1]);
	}
};

/*template<typename TBorderHandling>
struct LinearInterpolationImpl<1, TBorderHandling> {
	template<typename TView, typename TWeight, typename TIndex>
	static typename TView::value_type compute(TView view, TWeight weight, TIndex corner1, TIndex corner2) {
		return lerp(TBorderHandling::access(view, corner1, TIndex()), TBorderHandling::access(view, corner2, TIndex()), weight[0]);
	}
};*/

template<>
struct LinearInterpolationImpl<1> {
	template<typename TAccessor, typename TWeight, typename TIndex>
	CUGIP_DECL_HYBRID
	static typename TAccessor::value_type compute(TAccessor accessor, TWeight weight, TIndex corner1, TIndex corner2) {
		//std::cout << corner1 << "-" << corner2 << ";";
		return lerp(accessor[corner1], accessor[corner2], weight[0]);
	}
};

/*
template<>
struct LinearInterpolationImpl<2> {
	template<typename TView, typename TWeight, typename TIndex, typename TBorderHandling>
	static typename TView::Element compute(const TView &view, TWeight weight, TIndex corner1, TIndex corner2) {
		auto tmp_corner2 = corner2;
		tmp_corner2[1] = corner1[1];

		auto tmp_corner1 = corner1;
		tmp_corner1[1] = corner2[1];
		return Lerp(
			LinearInterpolationImpl<1>::Compute<TBorderHandling>(view, weight, corner1, tmp_corner2),
			LinearInterpolationImpl<1>::Compute<TBorderHandling>(view, weight, tmp_corner1, corner2),
			weight[1]);

	}
};

template<>
struct LinearInterpolationImpl<3> {

	template<typename TView, typename TWeight, typename TIndex, typename TBorderHandling>
	static typename TView::Element compute(const TView &view, TWeight weight, TIndex corner1, TIndex corner2) {
		auto tmp_corner2 = corner2;
		tmp_corner2[2] = corner1[2];

		auto tmp_corner1 = corner1;
		tmp_corner1[2] = corner2[2];
		return Lerp(
			LinearInterpolationImpl<2>::Compute<TBorderHandling>(view, weight, corner1, tmp_corner2),
			LinearInterpolationImpl<2>::Compute<TBorderHandling>(view, weight, tmp_corner1, corner2),
			weight[2]);

	}

};*/

}  // namespace detail

template<typename TBoundaryHandler = BorderHandlingTraits<border_handling_enum::REPEAT>>
struct NearestNeighborInterpolator {
	template<typename TView, typename TOffset, typename TIndex>
	CUGIP_DECL_HYBRID
	typename TView::value_type operator()(
			TView view,
			TOffset offset,
			TIndex index
			) const
	{
		return TBoundaryHandler::access(view, index, simple_vector<int, dimension<TView>::value>());
	}
};

template<typename TBoundaryHandler = BorderHandlingTraits<border_handling_enum::REPEAT>>
struct LinearInterpolator {
	CUGIP_HD_WARNING_DISABLE
	template<typename TView, typename TOffset, typename TIndex>
	CUGIP_DECL_HYBRID
	typename TView::value_type operator()(
			TView view,
			TOffset offset,
			TIndex index
			) const
	{
		/*auto floor = TView::IndexType(Floor(offset)) + index;
		auto ceil = TView::IndexType(Ceil(offset)) + index;
		auto weight = (offset + index) - floor;
		for (int i = 0; i < TView::kDimension; ++i) {

		}*/

		auto corner1 = TIndex(floor(index + offset));
		auto corner2 = TIndex(ceil(index + offset));
		auto weight = (index + offset) - corner1;
		//auto a = detail::LinearInterpolationImpl<dimension<TView>::value, TBoundaryHandler>();
		//return detail::LinearInterpolationImpl<dimension<TView>::value, TBoundaryHandler>::compute(view, weight, corner1, corner2);
		//std::cout << weight << ": ";
		auto value = detail::LinearInterpolationImpl<dimension<TView>::value>::compute(detail::make_accessor<TBoundaryHandler>(view), weight, corner1, corner2);
		//std::cout << "\n";
		return value;
		//return 0.0;
	}
};


/*
template<typename TImageView, typename TBorderHandling, typename TInterpolator>
class InterpolatedImageView: public TImageView
{
public:
	static const int kDimension = TImageView::kDimension;
	static const bool kIsDeviceView = TImageView::kIsDeviceView;
	static const bool kIsHostView = TImageView::kIsHostView;
	typedef typename TImageView::SizeType SizeType;
	typedef typename TImageView::IndexType IndexType;
	typedef typename TImageView::Element Element;
	typedef typename TImageView::AccessType AccessType;

	typedef Vector<float, tDimension> CoordinatesType;


	Element GetInterpolatedValue(CoordinatesType coordinates) const {
		auto rounded_coords = Round(coordinates);
		return interpolator_(*this, coordinates - rounded_coords, rounded_coords);
	}

protected:
	TInterpolator interpolator_;
};
*/


template<typename TView, typename TInterpolator = NearestNeighborInterpolator<BorderHandlingTraits<border_handling_enum::REPEAT>> >
class interpolated_view : public TView
{
public:
	static const int cDimension = dimension<TView>::value;
	typedef typename TView::value_type value_type;
	typedef typename dim_traits<cDimension>::extents_t extents_t;
	typedef typename dim_traits<cDimension>::coord_t coord_t;
	typedef typename dim_traits<cDimension>::diff_t diff_t;
	/*
	typedef const_host_image_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef const_value_type & accessed_type;*/
	typedef interpolated_view<TView, TInterpolator> this_t;
	typedef typename dim_traits<cDimension>::float_coord_t float_coord_t;

	CUGIP_DECL_HYBRID
	interpolated_view(TView aView)
		: TView(aView)
		, mOffset(0.5f, FillFlag())
	{}

	CUGIP_DECL_HYBRID
	interpolated_view(TView aView, TInterpolator aInterpolator)
		: TView(aView)
		, mInterpolator(aInterpolator)
		, mOffset(0.5f, FillFlag())
	{}

	CUGIP_DECL_HYBRID
	value_type interpolated_value(float_coord_t coordinates) const {
		// TODO where is 0 vs 0.0f
		//auto rounded_coords = round(coordinates);

		auto coordinatesIndex = coordinates - mOffset;
		auto rounded_coords = floor(coordinatesIndex);
		auto offset = coordinatesIndex - rounded_coords;
		//std::cout << "A " << coordinates << "; " << coordinatesIndex << "; " << rounded_coords << "; " << offset << "\n";
		return mInterpolator(*this, offset, rounded_coords);
	}

	CUGIP_DECL_HYBRID float_coord_t
	coordinates_from_index(coord_t aIndex) const
	{
		return aIndex + mOffset;
	}
protected:
	TInterpolator mInterpolator;
	float_coord_t mOffset;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(interpolated_view<TView, TInterpolator>),
	dimension<TView>::value,
	is_device_view<TView>::value,
	is_host_view<TView>::value,
	typename TView, typename TInterpolator);

template<typename TView, typename TInterpolator>
struct is_interpolated_view<interpolated_view<TView, TInterpolator>>: public std::true_type {};

template<typename TView>
interpolated_view<TView>
make_interpolated_view(TView aView)
{
	return interpolated_view<TView>(aView);
}

template<typename TView, typename TInterpolator>
interpolated_view<TView, TInterpolator>
make_interpolated_view(TView aView, TInterpolator aInterpolator)
{
	return interpolated_view<TView, TInterpolator>(aView, aInterpolator);
}

}  // namespace cugip
