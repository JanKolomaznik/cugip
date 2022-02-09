#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/detail/view_declaration_utils.hpp>
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>
#include <cugip/image_locator.hpp>
#include <cugip/math.hpp>

namespace cugip {

template<typename TElement, int tDim = 2>
class device_image_view
{
public:
	static const int cDimension = tDim;
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	typedef typename memory_management<TElement, tDim>::device_memory memory_t;
	typedef device_image_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef value_type & accessed_type;
	typedef device_ptr<TElement> pointer_t;

	/*device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}*/

	CUGIP_DECL_HYBRID
	device_image_view(TElement *buffer, extents_t size, extents_t strides)
		: mDevicePtr(buffer)
		, mSize(size)
		, mStrides(strides)
	{}

	device_image_view()
	{ /*empty*/ }

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{
		//return mData.dimensions();
		return mSize;
	}

	CUGIP_DECL_HYBRID accessed_type
	operator[](coord_t aCoords) const
	{
		CUGIP_ASSERT(aCoords >= extents_t());
		CUGIP_ASSERT(aCoords < this->dimensions());
		return *reinterpret_cast<value_type *>(reinterpret_cast<char *>(mDevicePtr) + offset_in_strided_memory(mStrides, aCoords));
		//return mData[aCoords];
	}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates) const
	{
		return image_locator<this_t, TBorderHandling>(*const_cast<this_t *>(this), aCoordinates); //TODO - remove const_cast
	}

	/*CUGIP_DECL_HYBRID const memory_t&
	data() const
	{ return mData; }*/

	CUGIP_DECL_HYBRID value_type *
	pointer() const
	{
		//return mData.mData.get();
		return mDevicePtr;
	}

	CUGIP_DECL_HYBRID extents_t
	strides() const
	{
		//return mData.strides();
		return mStrides;
	}

	device_image_view<TElement, tDim> subview(const coord_t &corner, const extents_t &size) const {
		// D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		CUGIP_ASSERT(corner >= extents_t());
		CUGIP_ASSERT(corner < this->dimensions());
		CUGIP_ASSERT((corner + size) <= this->dimensions());
		auto ptr = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mDevicePtr) + offset_in_strided_memory(mStrides, corner));
		//return device_image_view<TElement, tDimension>(this->mDevicePtr + linear_index_from_strides(this->mStrides, corner), size, this->strides_);
		return device_image_view<TElement, tDim>(ptr, size, this->mStrides);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	template<int tSliceDimension>
	device_image_view<TElement, cDimension - 1> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < cDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		CUGIP_ASSERT(slice >= 0);
		CUGIP_ASSERT(slice < this->dimensions()[tSliceDimension]);
		//TElement *slice_corner = this->mDevicePtr + int64_t(this->mStrides[tSliceDimension]) * slice;
		auto corner = coord_t();
		corner[tSliceDimension] = slice;
		value_type *slice_corner = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mDevicePtr) + offset_in_strided_memory(mStrides, corner));
		return device_image_view<TElement, cDimension - 1>(
				slice_corner,
				remove_dimension(this->mSize, tSliceDimension),
				remove_dimension(this->mStrides, tSliceDimension));
	}

protected:
	//memory_t mData;
	extents_t mSize;
	TElement *mDevicePtr;
	extents_t mStrides;
};

template<typename TElement, int tDim>
device_image_view<TElement, tDim>
view(const device_image_view<TElement, tDim> &aView)
{
	return aView;
}


template<typename TElement, int tDim = 2>
class const_device_image_view
{
public:
	static const int cDimension = tDim;
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	typedef typename memory_management<TElement, tDim>::const_device_memory memory_t;
	typedef const_device_image_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef const_value_type & accessed_type;
	typedef const_device_ptr<TElement> pointer_t;

	//friend struct cugip::detail::access_memory;

	/*CUGIP_DECL_HYBRID
	const_device_image_view(const typename memory_management<TElement, tDim>::const_device_memory &aData) :
		mData(aData)
	{}

	CUGIP_DECL_HYBRID
	const_device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}*/

	CUGIP_DECL_HYBRID
	const_device_image_view(const TElement *buffer, extents_t size, extents_t strides)
		: mDevicePtr(buffer)
		, mSize(size)
		, mStrides(strides)
	{}

	CUGIP_DECL_HYBRID
	const_device_image_view()
	{ /*empty*/ }

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{
		//return mData.dimensions();
		return mSize;
	}

	CUGIP_DECL_HYBRID accessed_type &
	operator[](coord_t aCoords) const
	{
		CUGIP_ASSERT(aCoords >= extents_t());
		CUGIP_ASSERT(aCoords < this->dimensions());
		return *reinterpret_cast<const_value_type *>(reinterpret_cast<const char *>(mDevicePtr) + offset_in_strided_memory(mStrides, aCoords));
	//	return mData[aCoords];
	}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates) const
	{
		return image_locator<this_t, TBorderHandling>(*const_cast<this_t *>(this), aCoordinates); //TODO - remove const_cast
	}

	/*CUGIP_DECL_HYBRID const memory_t&
	data() const
	{ return mData; }*/

	CUGIP_DECL_HYBRID const_value_type *
	pointer() const
	{
		//return mData.mData.get();
		return mDevicePtr;
	}

	CUGIP_DECL_HYBRID extents_t
	strides() const
	{
		//return mData.strides();
		return mStrides;
	}

	const_device_image_view<TElement, tDim> subview(const coord_t &corner, const extents_t &size) const {
		// D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		CUGIP_ASSERT(corner >= extents_t());
		CUGIP_ASSERT(corner < this->dimensions());
		CUGIP_ASSERT((corner + size) <= this->dimensions());
		auto ptr = reinterpret_cast<const value_type *>(reinterpret_cast<const char *>(mDevicePtr) + offset_in_strided_memory(mStrides, corner));
		//return device_image_view<TElement, tDimension>(this->mDevicePtr + linear_index_from_strides(this->mStrides, corner), size, this->strides_);
		return const_device_image_view<TElement, tDim>(ptr, size, this->mStrides);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	template<int tSliceDimension>
	const_device_image_view<TElement, cDimension - 1> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < cDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		CUGIP_ASSERT(slice >= 0);
		CUGIP_ASSERT(slice < this->dimensions()[tSliceDimension]);
		//TElement *slice_corner = this->mDevicePtr + int64_t(this->mStrides[tSliceDimension]) * slice;
		auto corner = coord_t();
		corner[tSliceDimension] = slice;
		const_value_type *slice_corner = reinterpret_cast<const_value_type *>(reinterpret_cast<const char *>(mDevicePtr) + offset_in_strided_memory(mStrides, corner));
		return const_device_image_view<TElement, cDimension - 1>(
				slice_corner,
				remove_dimension(this->mSize, tSliceDimension),
				remove_dimension(this->mStrides, tSliceDimension));
	}

protected:
	//memory_t mData;

	extents_t mSize;
	const TElement *mDevicePtr;
	extents_t mStrides;

};

template<typename TElement, int tDim>
const_device_image_view<TElement, tDim>
const_view(const device_image_view<TElement, tDim> &aView)
{
	return const_device_image_view<TElement, tDim>(aView.pointer(), aView.dimensions(), aView.strides());
}


template<typename TElement, int tDim>
const_device_image_view<TElement, tDim>
const_view(const const_device_image_view<TElement, tDim> &aView)
{
	return aView;
}


CUGIP_DECLARE_DEVICE_VIEW_TRAITS((device_image_view<TElement, tDim>), tDim, typename TElement, int tDim);
CUGIP_DECLARE_DEVICE_VIEW_TRAITS((const_device_image_view<TElement, tDim>), tDim, typename TElement, int tDim);

template<typename TElement, int tDim>
struct is_memory_based<device_image_view<TElement, tDim>>: public std::true_type {};

template<typename TElement, int tDim>
struct is_memory_based<const_device_image_view<TElement, tDim>>: public std::true_type {};

/*template<typename TElement, int tDim>
struct is_device_view<device_image_view<TElement, tDim> > : public std::true_type {};

template<typename TElement, int tDim>
struct is_device_view<const_device_image_view<TElement, tDim> > : public std::true_type {};

template<typename TElement, int tDim>
struct dimension<device_image_view<TElement, tDim> >: dimension_helper<tDim> {};

template<typename TElement, int tDim>
struct dimension<const_device_image_view<TElement, tDim> >: dimension_helper<tDim> {};*/

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID const_device_image_view<const TElement, tDimension>
makeConstDeviceImageView(const TElement *buffer, simple_vector<int, tDimension> size) {
	return const_device_image_view<const TElement, tDimension>(buffer, size, sizeof(TElement) * stridesFromSize(size));
}

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID const_device_image_view<const TElement, tDimension>
makeConstDeviceImageView(const TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return const_device_image_view<const TElement, tDimension>(buffer, size, strides);
}


template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID device_image_view<TElement, tDimension>
makeDeviceImageView(TElement *buffer, simple_vector<int, tDimension> size) {
	return device_image_view<TElement, tDimension>(buffer, size, sizeof(TElement) * stridesFromSize(size));
}

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID device_image_view<TElement, tDimension>
makeDeviceImageView(TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return device_image_view<TElement, tDimension>(buffer, size, strides);
}

/*template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID const_device_image_view<const TElement, tDimension>
makeConstDeviceImageView(const TElement *buffer, simple_vector<int, tDimension> size) {
	return const_device_image_view<const TElement, tDimension>(typename memory_management<TElement, tDimension>::const_device_memory(buffer, size, sizeof(TElement) * stridesFromSize(size)));
}

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID const_device_image_view<const TElement, tDimension>
makeConstDeviceImageView(const TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return const_device_image_view<const TElement, tDimension>(typename memory_management<TElement, tDimension>::const_device_memory(buffer, size, strides));
}


template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID device_image_view<TElement, tDimension>
makeDeviceImageView(TElement *buffer, simple_vector<int, tDimension> size) {
	return device_image_view<TElement, tDimension>(typename memory_management<TElement, tDimension>::device_memory(buffer, size, sizeof(TElement) * stridesFromSize(size)));
}

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID device_image_view<TElement, tDimension>
makeDeviceImageView(TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return device_image_view<TElement, tDimension>(typename memory_management<TElement, tDimension>::device_memory(buffer, size, strides));
}*/


/**
 * Create image view from raw array.
 **/
template<typename TElement>
CUGIP_DECL_HYBRID device_image_view<TElement, 2>
view(device_ptr<TElement> aData, typename dim_traits<2>::extents_t aExtents, size_t aPitch)
{
	CUGIP_ASSERT(aData);
	return device_image_view<TElement, 2>(typename memory_management<TElement, 2>::device_memory(aData, aExtents, aPitch));
}

/**
 * Create image view from raw array.
 **/
template<typename TElement>
CUGIP_DECL_HYBRID device_image_view<TElement, 2>
view(device_ptr<TElement> aData, typename dim_traits<2>::extents_t aExtents)
{
	CUGIP_ASSERT(aData);
	return device_image_view<TElement, 2>(
			typename memory_management<TElement, 2>::device_memory(aData, aExtents, aExtents[0]*sizeof(TElement)));
}

/**
 * Create const image view from raw array.
 **/
template<typename TElement, size_t tDim>
CUGIP_DECL_HYBRID const_device_image_view<TElement, tDim>
const_view(const_device_ptr<TElement> aData, typename dim_traits<tDim>::extents_t aExtents, size_t aPitch)
{
	CUGIP_ASSERT(aData);
	return const_device_image_view<TElement, tDim>(typename memory_management<TElement, tDim>::const_device_memory(aData, aExtents, aPitch));
}

/**
 * Create image view from raw array.
 **/
/*template<typename TImageView>
CUGIP_DECL_HYBRID TImageView
sub_image_view(TImageView aImage, typename TImageView::coord_t aCorner, typename TImageView::extents_t aExtents)
{
	CUGIP_ASSERT(!cugip::less(aImage.dimensions(), aCorner + aExtents));
	typename TImageView::pointer_t ptr = &aImage[aCorner];
	return TImageView(typename TImageView::memory_t(ptr, aExtents, detail::access_memory::get(aImage).mPitch));
}*/

}//namespace cugip
