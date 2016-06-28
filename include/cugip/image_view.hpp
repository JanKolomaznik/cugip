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

	device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}

	device_image_view()
	{ /*empty*/ }

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mData.dimensions(); }

	CUGIP_DECL_HYBRID accessed_type
	operator[](coord_t aCoords) const
	{
		return mData[aCoords];
	}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates) const
	{
		return image_locator<this_t, TBorderHandling>(*const_cast<this_t *>(this), aCoordinates); //TODO - remove const_cast
	}

	CUGIP_DECL_HYBRID const memory_t&
	data() const
	{ return mData; }

	value_type *
	pointer() const
	{
		return mData.mData.get();
	}

	extents_t
	strides() const
	{
		return mData.strides();
	}

protected:
	memory_t mData;

};

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

	const_device_image_view(const typename memory_management<TElement, tDim>::const_device_memory &aData) :
		mData(aData)
	{}

	const_device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}

	const_device_image_view()
	{ /*empty*/ }

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mData.dimensions(); }

	CUGIP_DECL_HYBRID accessed_type &
	operator[](coord_t aCoords) const
	{
		return mData[aCoords];
	}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates) const
	{
		return image_locator<this_t, TBorderHandling>(*const_cast<this_t *>(this), aCoordinates); //TODO - remove const_cast
	}

	CUGIP_DECL_HYBRID const memory_t&
	data() const
	{ return mData; }

	const_value_type *
	pointer() const
	{
		return mData.mData.get();
	}

	extents_t
	strides() const
	{
		return mData.strides();
	}
protected:
	memory_t mData;
};

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


}//namespace cugip
