// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <type_traits>

#include <cugip/detail/view_declaration_utils.hpp>

namespace cugip {

/// \addtogroup Views
/// @{


template<typename TView>
class subimage_view : public hybrid_image_view_base<dimension<TView>::value> {
public:
	typedef hybrid_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	CUGIP_DECL_HYBRID
	subimage_view(TView aView, const coord_t &corner, const extents_t &size) :
		predecessor_type(size),
		view_(aView),
		corner_(corner)
	{
	}

	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		index += corner_;
		return view_[index];
	}

protected:
	TView view_;
	coord_t corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};

CUGIP_DECLARE_VIEW_TRAITS(
	(subimage_view<TView>),
	dimension<TView>::value,
	is_device_view<TView>::value,
	is_host_view<TView>::value,
	typename TView);


template<typename TView>
class bordered_subimage_view : public hybrid_image_view_base<dimension<TView>::value> {
public:
	typedef hybrid_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	CUGIP_DECL_HYBRID
	bordered_subimage_view(TView aView, const coord_t &corner, const extents_t &size) :
		predecessor_type(size),
		view_(aView),
		corner_(corner)
	{
	}

	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		index += corner_;
		return view_[index];
	}

	CUGIP_DECL_HYBRID
	const TView &parent_view() const {
		return view_;
	}

	CUGIP_DECL_HYBRID
	const coord_t &corner() const {
		return corner_;
	}

protected:
	TView view_;
	coord_t corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};

CUGIP_DECLARE_VIEW_TRAITS(
	(bordered_subimage_view<TView>),
	dimension<TView>::value,
	is_device_view<TView>::value,
	is_host_view<TView>::value,
	typename TView);


CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID
region<dimension<TView>::value> valid_region(const bordered_subimage_view<TView> &aView) {
	auto region = valid_region(aView.parent_view());
	region.corner -= aView.corner();
	return region;
}

/*
/// Wrapper providing access to the cut through the wrapped view.
/// \tparam tSliceDimension Which axis is perpendicular to the cut.
template<typename TView, int tSliceDimension, bool tIsDeviceView>
class slice_image_view;

#if defined(__CUDACC__)
template<typename TView, int tSliceDimension>
class slice_image_view<TView, tSliceDimension, true> : public device_image_view_base<dimension<TView>::value - 1> {
public:
	typedef device_image_view_base<dimension<TView>::value - 1> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	slice_image_view(TView aView, int slice) :
		predecessor_type(RemoveDimension(aView.Size(), tSliceDimension)),
		view_(aView),
		slice_coordinate_(slice)
	{}

	CUGIP_DECL_DEVICE
	accessed_type operator[](coord_t index) const {
		auto new_index = InsertDimension(index, slice_coordinate_, tSliceDimension);
		return view_[new_index];
	}

protected:
	TView view_;
	int slice_coordinate_;
};
#endif  // __CUDACC__
*/
template<typename TView, int tSliceDimension>
class slice_image_view : public hybrid_image_view_base<dimension<TView>::value - 1> {
public:
	typedef hybrid_image_view_base<dimension<TView>::value - 1> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value - 1)

	CUGIP_DECL_HYBRID
	slice_image_view(TView aView, int slice) :
		predecessor_type(remove_dimension(aView.dimensions(), tSliceDimension)),
		view_(aView),
		slice_coordinate_(slice)
	{}

	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		auto new_index = insert_dimension(index, slice_coordinate_, tSliceDimension);
		return view_[new_index];
	}

protected:
	TView view_;
	int slice_coordinate_;
};


CUGIP_DECLARE_VIEW_TRAITS(
	(slice_image_view<TView, tSliceDimension>),
	dimension<TView>::value,
	is_device_view<TView>::value,
	is_host_view<TView>::value,
	typename TView, int tSliceDimension);


template<typename TView>
class strided_subimage_view : public hybrid_image_view_base<dimension<TView>::value> {
public:
	typedef hybrid_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	CUGIP_DECL_HYBRID
	strided_subimage_view(TView aView, coord_t aOffset, coord_t aStrides) :
		predecessor_type(div(aView.dimensions() - aOffset, aStrides)),
		mView(aView),
		mOffset(aOffset),
		mStrides(aStrides)
	{}

	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return mView[mOffset + product(index, mStrides)];
	}

protected:
	TView mView;
	coord_t mOffset;
	coord_t mStrides;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(strided_subimage_view<TView>),
	dimension<TView>::value,
	is_device_view<TView>::value,
	is_host_view<TView>::value,
	typename TView);




namespace detail {

template<typename TView, bool tIsMemoryBased>
struct SubviewGenerator {
	typedef subimage_view<TView> ResultView;

	static ResultView invoke(
		TView aView,
		const simple_vector<int, dimension<TView>::value> &corner,
		const simple_vector<int, dimension<TView>::value> &size)
	{
		return ResultView(aView, corner, size);
	}
};


template<typename TView>
struct SubviewGenerator<TView, true> {
	typedef TView ResultView;

	static ResultView invoke(
		TView aView,
		const simple_vector<int, dimension<TView>::value> &corner,
		const simple_vector<int, dimension<TView>::value> &size)
	{
		return aView.subview(corner, size);
	}
};


template<typename TView, int tSliceDimension, bool tIsMemoryBased>
struct SliceGenerator {
	typedef slice_image_view<TView, tSliceDimension> ResultView;

	static ResultView invoke(TView aView, int slice) {
		return ResultView(aView, slice);
	}
};


template<typename TView, int tSliceDimension>
struct SliceGenerator<TView, tSliceDimension, true> {
	typedef decltype(std::declval<TView>().template slice<tSliceDimension>(0)) ResultView;

	static ResultView invoke(TView aView, int slice) {
		return aView.template slice<tSliceDimension>(slice);
	}
};

}  // namespace detail

/// Creates view for part of the original image view (device or host).
/// When the original view is not memory based it returns view wrapper.
/// \param view Original image view.
/// \param corner Index of view corner (zero coordinates in new view)
/// \param size Size of the subview
template<typename TView>
auto subview(
	const TView &aView,
	const simple_vector<int, dimension<TView>::value> &corner,
	const simple_vector<int, dimension<TView>::value> &size)
	-> typename detail::SubviewGenerator<TView, is_memory_based<TView>::value>::ResultView
{
	static_assert(is_image_view<TView>::value, "Subview can be generated only for image views.");
	//D_FORMAT("Generating subview: corner: %1%, size: %2%, original size: %3%", corner, size, view.Size());
	//CUGIP_ASSERT((corner >= Vector<int, TView::kDimension>()));
	//CUGIP_ASSERT(corner < aView.Size());
	//CUGIP_ASSERT((corner + size) <= aView.Size());
	bool cornerInside = corner >= simple_vector<int, dimension<TView>::value>() && corner < aView.dimensions();
	if (!cornerInside || !((corner + size) <= aView.dimensions())) {
		CUGIP_THROW(EInvalidRange() << sourceRegionErrorInfo(active_region(aView)) << targetRegionErrorInfo(region<dimension<TView>::value>{corner, size}));
	}
	return detail::SubviewGenerator<TView, is_memory_based<TView>::value>::invoke(aView, corner, size);
}

/// Creates view for part of the original image view (device or host).
/// In behavior same as the normal subview, but read access to elements
/// outside its domain is valid, as long as it is valid in the original image view.
/// \param view Original image view.
/// \param corner Index of view corner (zero coordinates in new view)
/// \param size Size of the subview
template<typename TView>
auto bordered_subview(
	const TView &aView,
	const simple_vector<int, dimension<TView>::value> &corner,
	const simple_vector<int, dimension<TView>::value> &size)
	-> bordered_subimage_view<TView>
{
	static_assert(is_image_view<TView>::value, "Subview can be generated only for image views.");
	//D_FORMAT("Generating subview: corner: %1%, size: %2%, original size: %3%", corner, size, aView.Size());
	//CUGIP_ASSERT((corner >= Vector<int, TView::kDimension>()));
	//CUGIP_ASSERT(corner < aView.Size());
	//CUGIP_ASSERT((corner + size) <= aView.Size());
	bool cornerInside = corner >= simple_vector<int, dimension<TView>::value>() && corner < aView.dimensions();
	if (!cornerInside || !((corner + size) <= aView.dimensions())) {
		CUGIP_THROW(EInvalidRange() << sourceRegionErrorInfo(active_region(aView)) << targetRegionErrorInfo(region<dimension<TView>::value>{corner, size}));
	}
	return bordered_subimage_view<TView>(aView, corner, size);
}


/// Creates slice view (view of smaller dimension).
/// \tparam tSliceDimension Which axis is perpendicular to the cut
/// TODO(johny) - generic slicing
template<int tSliceDimension, typename TView>
auto slice(
	const TView &aView,
	int slice)
	-> typename detail::SliceGenerator<TView, tSliceDimension, is_memory_based<TView>::value>::ResultView
{
	static_assert(is_image_view<TView>::value, "Only image view can be sliced.");
	CUGIP_ASSERT(slice >= 0);
	CUGIP_ASSERT(slice < aView.dimensions()[tSliceDimension]);
	if (slice < 0 || slice >= aView.dimensions()[tSliceDimension]) {
		CUGIP_THROW(ESliceOutOfRange() << sourceRegionErrorInfo(active_region(aView)) << InvalidSliceErrorInfo(vect2i_t{slice, tSliceDimension}));
	}
	return detail::SliceGenerator<TView, tSliceDimension, is_memory_based<TView>::value>::invoke(aView, slice);
}


template<typename TView>
auto strided_subview(
	const TView &aView,
	typename TView::coord_t aOffset,
	typename TView::coord_t aStrides)
	-> strided_subimage_view<TView>
{
	static_assert(is_image_view<TView>::value, "Subview can be generated only for image views.");
	//TODO memory based
	return strided_subimage_view<TView>(aView, aOffset, aStrides);
}


/// @}

}  // namespace cugip
