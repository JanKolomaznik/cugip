#pragma once

#if ! defined(__CUDACC__)
#error "This header can be included only into sources compiled by nvcc."
#endif  // !defined(__CUDACC__)

#include <cugip/detail/include.hpp>
#include <cugip/detail/view_declaration_utils.hpp>
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>
#include <cugip/image_locator.hpp>
#include <cugip/math.hpp>

namespace cugip {

/// \addtogroup Views
/// @{

/// View to the part or whole unified-memory image, which owns the data.
/// It provides only constant access to the data. It is usable on both host/device sides.
template<typename TElement, int tDim>
class const_unified_image_view
{
public:
	static const int cDimension = tDim;
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	//typedef typename memory_management<TElement, tDim>::unified_memory memory_t;
	typedef const_unified_image_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef const_value_type & accessed_type;

	CUGIP_DECL_HYBRID
	const_unified_image_view(const TElement *buffer, extents_t size, extents_t strides)
		: mUnifiedPtr(buffer)
		, mSize(size)
		, mStrides(strides)
	{}

	CUGIP_DECL_HYBRID
	const_unified_image_view()
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
		return *reinterpret_cast<const_value_type *>(reinterpret_cast<const char *>(mUnifiedPtr) + offset_in_strided_memory(mStrides, aCoords));
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
		return mUnifiedPtr;
	}

	CUGIP_DECL_HYBRID extents_t
	strides() const
	{
		//return mData.strides();
		return mStrides;
	}

	const_unified_image_view<TElement, tDim> subview(const coord_t &corner, const extents_t &size) const {
		// D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		CUGIP_ASSERT(corner >= extents_t());
		CUGIP_ASSERT(corner < this->dimensions());
		CUGIP_ASSERT((corner + size) <= this->dimensions());
		auto ptr = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mUnifiedPtr) + offset_in_strided_memory(mStrides, corner));
		//return unified_image_view<TElement, tDimension>(this->mDevicePtr + linear_index_from_strides(this->mStrides, corner), size, this->strides_);
		return const_unified_image_view<TElement, tDim>(ptr, size, this->mStrides);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	/*template<int tSliceDimension>
	unified_image_view<TElement, tDimension - 1> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		CUGIP_ASSERT(slice >= 0);
		CUGIP_ASSERT(slice < this->dimensions()[tSliceDimension]);
		TElement *slice_corner = this->mDevicePtr + int64_t(this->mStrides[tSliceDimension]) * slice;
		return unified_image_view<TElement, tDimension - 1>(
				slice_corner,
				RemoveDimension(this->size_, tSliceDimension),
				RemoveDimension(this->mStrides, tSliceDimension));
	}*/

protected:
	extents_t mSize;
	const TElement *mUnifiedPtr;
	extents_t mStrides;
};


/// View to the part or whole unified-memory image, which owns the data.
/// It provides reference access to the data. It is usable on both host/device sides.
template<typename TElement, int tDim>
class unified_image_view
{
public:
	static const int cDimension = tDim;
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	//typedef typename memory_management<TElement, tDim>::const_unified_memory memory_t;
	typedef const_unified_image_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef value_type & accessed_type;

	CUGIP_DECL_HYBRID
	unified_image_view(TElement *buffer, extents_t size, extents_t strides)
		: mUnifiedPtr(buffer)
		, mSize(size)
		, mStrides(strides)
	{}

	CUGIP_DECL_HYBRID
	unified_image_view()
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
		return *reinterpret_cast<value_type *>(reinterpret_cast<char *>(mUnifiedPtr) + offset_in_strided_memory(mStrides, aCoords));
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
		return mUnifiedPtr;
	}

	CUGIP_DECL_HYBRID extents_t
	strides() const
	{
		//return mData.strides();
		return mStrides;
	}

	unified_image_view<TElement, tDim> subview(const coord_t &corner, const extents_t &size) const {
		// D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		CUGIP_ASSERT(corner >= extents_t());
		CUGIP_ASSERT(corner < this->dimensions());
		CUGIP_ASSERT((corner + size) <= this->dimensions());
		auto ptr = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mUnifiedPtr) + offset_in_strided_memory(mStrides, corner));
		//return unified_image_view<TElement, tDimension>(this->mDevicePtr + linear_index_from_strides(this->mStrides, corner), size, this->strides_);
		return unified_image_view<TElement, tDim>(ptr, size, this->mStrides);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	/*template<int tSliceDimension>
	unified_image_view<TElement, tDimension - 1> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		CUGIP_ASSERT(slice >= 0);
		CUGIP_ASSERT(slice < this->dimensions()[tSliceDimension]);
		TElement *slice_corner = this->mDevicePtr + int64_t(this->mStrides[tSliceDimension]) * slice;
		return unified_image_view<TElement, tDimension - 1>(
				slice_corner,
				RemoveDimension(this->size_, tSliceDimension),
				RemoveDimension(this->mStrides, tSliceDimension));
	}*/

protected:
	extents_t mSize;
	TElement *mUnifiedPtr;
	extents_t mStrides;
};

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((unified_image_view<TElement, tDim>), tDim, typename TElement, int tDim);
CUGIP_DECLARE_HYBRID_VIEW_TRAITS((const_unified_image_view<TElement, tDim>), tDim, typename TElement, int tDim);

template<typename TElement, int tDim>
struct is_memory_based<unified_image_view<TElement, tDim>>: public std::true_type {};

template<typename TElement, int tDim>
struct is_memory_based<const_unified_image_view<TElement, tDim>>: public std::true_type {};

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID const_unified_image_view<const TElement, tDimension>
makeConstUnifiedImageView(const TElement *buffer, simple_vector<int, tDimension> size) {
	return const_unified_image_view<const TElement, tDimension>(buffer, size, sizeof(TElement) * stridesFromSize(size));
}

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID const_unified_image_view<const TElement, tDimension>
makeConstUnifiedImageView(const TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return const_unified_image_view<const TElement, tDimension>(buffer, size, strides);
}


template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID unified_image_view<TElement, tDimension>
makeUnifiedImageView(TElement *buffer, simple_vector<int, tDimension> size) {
	return unified_image_view<TElement, tDimension>(buffer, size, sizeof(TElement) * stridesFromSize(size));
}

template<typename TElement, int tDimension>
CUGIP_DECL_HYBRID unified_image_view<TElement, tDimension>
makeUnifiedImageView(TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return unified_image_view<TElement, tDimension>(buffer, size, strides);
}


/*
template<typename TView>
void prefetch_view(TView aView, int aDevice, cudaStream_t aStream = 0){
	int elementSize = sizeof(*(view.Pointer()));

	if(view.HasContiguousMemory()){
		cudaMemPrefetchAsync(
				aView.Pointer(),
				aView.ElementCount()*elementSize,
				aDevice,
				aStream);
	}

	// Non contiguous memory prefetch too slow

	// else{
	// 	const int columns = view.Size()[0];
	// 	const int lines = view.Size()[1];
	// 	const int stride = view.Strides()[1];
	//
	// 	for(int i = 0; i < lines; ++i){
	// 		cudaMemPrefetchAsync(
	// 			view.Pointer() + i*stride,
	// 			columns * elementSize,
	// 			dstDevice,
	// 			stream);
	// 	}
	// }
}*/

/// @}

}//namespace cugip
