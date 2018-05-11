#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/detail/view_declaration_utils.hpp>
#include <cugip/utils.hpp>
#include <cugip/access_utils.hpp>
//#include <cugip/memory.hpp>
//#include <cugip/image_locator.hpp>
#include <cugip/math.hpp>

namespace cugip {

template<typename TElement, int tDim = 2>
class host_image_view
{
public:
	static const int cDimension = tDim;
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	typedef host_image_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef value_type & accessed_type;

	host_image_view(TElement *host_ptr, extents_t size, extents_t strides)
		: mSize(size)
		, mHostPtr(host_ptr)
		, mStrides(strides)
	{}

	extents_t
	dimensions() const
	{ return mSize; }

	accessed_type
	operator[](coord_t aCoords) const
	{
		return *reinterpret_cast<value_type *>(reinterpret_cast<char *>(mHostPtr) + offset_in_strided_memory(mStrides, aCoords));
		//return mHostPtr[dot(mStrides, aCoords)];
	}

	//TODO - locators for host views
	/*template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates)
	{
		return image_locator<this_t, TBorderHandling>(*this, aCoordinates);
	}*/

	value_type *
	pointer() const
	{
		return mHostPtr;
	}

	extents_t
	strides() const
	{
		return mStrides;
	}

protected:
	extents_t mSize;
	TElement *mHostPtr;
	extents_t mStrides;
};

template<typename TElement, int tDim = 2>
class const_host_image_view
{
public:
	static const int cDimension = tDim;
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	typedef const_host_image_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef const_value_type & accessed_type;

	const_host_image_view(TElement *host_ptr, extents_t size, extents_t strides)
		: mSize(size)
		, mHostPtr(host_ptr)
		, mStrides(strides)
	{}

	extents_t
	dimensions() const
	{ return mSize; }

	accessed_type
	operator[](coord_t aCoords) const
	{
		return *reinterpret_cast<const_value_type *>(reinterpret_cast<const char *>(mHostPtr) + offset_in_strided_memory(mStrides, aCoords));
		//return mHostPtr[dot(mStrides, aCoords)];
	}


	/*template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates)
	{
		return image_locator<this_t, TBorderHandling>(*this, aCoordinates);
	}*/

	value_type *
	pointer() const
	{
		return mHostPtr;
	}

	extents_t
	strides() const
	{
		return mStrides;
	}

protected:
	extents_t mSize;
	TElement *mHostPtr;
	extents_t mStrides;
};

CUGIP_DECLARE_HOST_VIEW_TRAITS((host_image_view<TElement, tDim>), tDim, typename TElement, int tDim);
CUGIP_DECLARE_HOST_VIEW_TRAITS((const_host_image_view<TElement, tDim>), tDim, typename TElement, int tDim);

template<typename TElement, int tDim>
struct is_memory_based<host_image_view<TElement, tDim>>: public std::true_type {};

template<typename TElement, int tDim>
struct is_memory_based<const_host_image_view<TElement, tDim>>: public std::true_type {};

template<typename TElement, int tDimension>
const_host_image_view<const TElement, tDimension>
makeConstHostImageView(const TElement *buffer, simple_vector<int, tDimension> size) {
	return const_host_image_view<const TElement, tDimension>(buffer, size, sizeof(TElement) * stridesFromSize(size));
}

template<typename TElement, int tDimension>
const_host_image_view<const TElement, tDimension>
makeConstHostImageView(const TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return const_host_image_view<const TElement, tDimension>(buffer, size, strides);
}


template<typename TElement, int tDimension>
host_image_view<TElement, tDimension>
makeHostImageView(TElement *buffer, simple_vector<int, tDimension> size) {
	return host_image_view<TElement, tDimension>(buffer, size, sizeof(TElement) * stridesFromSize(size));
}

template<typename TElement, int tDimension>
host_image_view<TElement, tDimension>
makeHostImageView(TElement *buffer, simple_vector<int, tDimension> size, simple_vector<int, tDimension> strides) {
	return host_image_view<TElement, tDimension>(buffer, size, strides);
}

}//namespace cugip
