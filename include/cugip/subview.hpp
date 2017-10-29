// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <type_traits>
#include <cugip/detail/view_declaration_utils.hpp>

namespace cugip {

/// \addtogroup Views
/// @{

/// Wrapper limiting access to the wrapped view
template<typename TView, bool tIsDeviceView>
class subimage_view;

template<typename TView, bool tIsDeviceView>
class bordered_subimage_view;

#if defined(__CUDACC__)
template<typename TView>
class subimage_view<TView, true> : public device_image_view_base<dimension<TView>::value> {
public:
	typedef device_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	subimage_view(TView view, const coord_t &corner, const extents_t &size) :
		predecessor_type(size),
		view_(view),
		corner_(corner)
	{
		static_assert(is_device_view<TView>::value && !is_host_view<TView>::value, "Only pure device views currently supported.");
	}

	CUGIP_DECL_DEVICE
	accessed_type operator[](coord_t index) const {
		index += corner_;
		return view_[index];
	}

protected:
	TView view_;
	coord_t corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};
#endif  // __CUDACC__

template<typename TView>
class subimage_view<TView, false> : public host_image_view_base<dimension<TView>::value> {
public:
	typedef host_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)


	subimage_view(TView view, const coord_t &corner, const extents_t &size) :
		predecessor_type(size),
		view_(view),
		corner_(corner)
	{
		static_assert(!is_device_view<TView>::value && is_host_view<TView>::value, "Only pure host views currently supported.");
	}

	accessed_type operator[](coord_t index) const {
		index += corner_;
		return view_[index];
	}

protected:
	TView view_;
	coord_t corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};

CUGIP_DECLARE_VIEW_TRAITS(
	(subimage_view<TView, tIsDeviceView>),
	dimension<TView>::value,
	tIsDeviceView,
	(!tIsDeviceView),
	typename TView, bool tIsDeviceView);


#if defined(__CUDACC__)
template<typename TView>
class bordered_subimage_view<TView, true> : public device_image_view_base<dimension<TView>::value> {
public:
	typedef device_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	bordered_subimage_view(TView view, const coord_t &corner, const extents_t &size) :
		predecessor_type(size),
		view_(view),
		corner_(corner)
	{
		static_assert(is_device_view<TView>::value && !is_host_view<TView>::value, "Only pure device views currently supported.");
	}

	CUGIP_DECL_DEVICE
	accessed_type operator[](coord_t index) const {
		index += corner_;
		return view_[index];
	}

	CUGIP_DECL_DEVICE
	const TView &parent_view() const {
		return view_;
	}

	CUGIP_DECL_DEVICE
	const coord_t &corner() const {
		return corner_;
	}

protected:
	TView view_;
	coord_t corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};
#endif  // __CUDACC__

template<typename TView>
class bordered_subimage_view<TView, false> : public host_image_view_base<dimension<TView>::value> {
public:
	typedef host_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	bordered_subimage_view(TView view, const coord_t &corner, const extents_t &size) :
		predecessor_type(size),
		view_(view),
		corner_(corner)
	{
		static_assert(!is_device_view<TView>::value && is_host_view<TView>::value, "Only pure host views currently supported.");
	}

	accessed_type operator[](coord_t index) const {
		index += corner_;
		return view_[index];
	}

	const TView &parent_view() const {
		return view_;
	}

	const coord_t &corner() const {
		return corner_;
	}

protected:
	TView view_;
	coord_t corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};

CUGIP_DECLARE_VIEW_TRAITS(
	(bordered_subimage_view<TView, tIsDeviceView>),
	dimension<TView>::value,
	tIsDeviceView,
	(!tIsDeviceView),
	typename TView, bool tIsDeviceView);


CUGIP_HD_WARNING_DISABLE
template<typename TView, bool tIsDeviceView>
CUGIP_DECL_HYBRID
region<dimension<TView>::value> valid_region(const bordered_subimage_view<TView, tIsDeviceView> &view) {
	auto region = valid_region(view.parent_view());
	region.corner -= view.corner();
	return region;
}


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

	slice_image_view(TView view, int slice) :
		predecessor_type(RemoveDimension(view.Size(), tSliceDimension)),
		view_(view),
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

template<typename TView, int tSliceDimension>
class slice_image_view<TView, tSliceDimension, false> : public host_image_view_base<dimension<TView>::value - 1> {
public:
	typedef host_image_view_base<dimension<TView>::value - 1> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	slice_image_view(TView view, int slice) :
		predecessor_type(RemoveDimension(view.Size(), tSliceDimension)),
		view_(view),
		slice_coordinate_(slice)
	{}

	accessed_type operator[](coord_t index) const {
		auto new_index = InsertDimension(index, slice_coordinate_, tSliceDimension);
		return view_[new_index];
	}

protected:
	TView view_;
	int slice_coordinate_;
};


namespace detail {

template<typename TView, bool tIsMemoryBased>
struct SubviewGenerator {
	typedef subimage_view<TView, is_device_view<TView>::value> ResultView;

	static ResultView invoke(
		TView view,
		const simple_vector<int, dimension<TView>::value> &corner,
		const simple_vector<int, dimension<TView>::value> &size)
		/*const typename TView::IndexType &corner,
		const typename TView::SizeType &size)*/
	{
		return ResultView(view, corner, size);
	}
};


template<typename TView>
struct SubviewGenerator<TView, true> {
	typedef TView ResultView;

	static ResultView invoke(
		TView view,
		const simple_vector<int, dimension<TView>::value> &corner,
		const simple_vector<int, dimension<TView>::value> &size)
		/*const typename TView::IndexType &corner,
		const typename TView::SizeType &size)*/
	{
		return view.subview(corner, size);
	}
};


template<typename TView, int tSliceDimension, bool tIsMemoryBased>
struct SliceGenerator {
	typedef slice_image_view<TView, tSliceDimension, is_device_view<TView>::value> ResultView;

	static ResultView invoke(TView view, int slice) {
		return ResultView(view, slice);
	}
};


template<typename TView, int tSliceDimension>
struct SliceGenerator<TView, tSliceDimension, true> {
	typedef decltype(std::declval<TView>().template Slice<tSliceDimension>(0)) ResultView;

	static ResultView invoke(TView view, int slice) {
		return view.template Slice<tSliceDimension>(slice);
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
	const TView &view,
	const simple_vector<int, dimension<TView>::value> &corner,
	const simple_vector<int, dimension<TView>::value> &size)
	-> typename detail::SubviewGenerator<TView, is_memory_based<TView>::value>::ResultView
{
	//D_FORMAT("Generating subview: corner: %1%, size: %2%, original size: %3%", corner, size, view.Size());
	//CUGIP_ASSERT((corner >= Vector<int, TView::kDimension>()));
	//CUGIP_ASSERT(corner < view.Size());
	//CUGIP_ASSERT((corner + size) <= view.Size());
	bool cornerInside = corner >= simple_vector<int, dimension<TView>::value>() && corner < view.dimensions();
	if (!cornerInside || !((corner + size) <= view.dimensions())) {
		CUGIP_THROW(100);
		//CUGIP_THROW(InvalidNDRange() << GetOriginalRegionErrorInfo(view.GetRegion()) << GetWrongRegionErrorInfo(CreateRegion(corner, size)));
	}
	return detail::SubviewGenerator<TView, is_memory_based<TView>::value>::invoke(view, corner, size);
}

/// Creates view for part of the original image view (device or host).
/// In behavior same as the normal subview, but read access to elements
/// outside its domain is valid, as long as it is valid in the original image view.
/// \param view Original image view.
/// \param corner Index of view corner (zero coordinates in new view)
/// \param size Size of the subview
template<typename TView>
auto bordered_subview(
	const TView &view,
	const simple_vector<int, dimension<TView>::value> &corner,
	const simple_vector<int, dimension<TView>::value> &size)
	-> bordered_subimage_view<TView, is_device_view<TView>::value>
{
	//D_FORMAT("Generating subview: corner: %1%, size: %2%, original size: %3%", corner, size, view.Size());
	//CUGIP_ASSERT((corner >= Vector<int, TView::kDimension>()));
	//CUGIP_ASSERT(corner < view.Size());
	//CUGIP_ASSERT((corner + size) <= view.Size());
	bool cornerInside = corner >= simple_vector<int, dimension<TView>::value>() && corner < view.dimensions();
	if (!cornerInside || !((corner + size) <= view.dimensions())) {
		CUGIP_THROW(100);
		//CUGIP_THROW(InvalidNDRange() << GetOriginalRegionErrorInfo(view.GetRegion()) << GetWrongRegionErrorInfo(CreateRegion(corner, size)));
	}
	return bordered_subimage_view<TView, is_device_view<TView>::value>(view, corner, size);
}


/// Creates slice view (view of smaller dimension).
/// \tparam tSliceDimension Which axis is perpendicular to the cut
/// TODO(johny) - generic slicing
template<int tSliceDimension, typename TView>
auto slice(
	const TView &view,
	int slice)
	-> typename detail::SliceGenerator<TView, tSliceDimension, is_memory_based<TView>::value>::ResultView
{
	CUGIP_ASSERT(slice >= 0);
	CUGIP_ASSERT(slice < view.dimensions()[tSliceDimension]);
	if (slice < 0 || slice >= view.dimensions()[tSliceDimension]) {
		CUGIP_THROW(100);
		//CUGIP_THROW(SliceOutOfRange() << GetOriginalRegionErrorInfo(view.GetRegion()) << WrongSliceErrorInfo(Int2(slice, tSliceDimension)));
	}
	return detail::SliceGenerator<TView, tSliceDimension, is_memory_based<TView>::value>::invoke(view, slice);
}

/// @}

}  // namespace cugip
