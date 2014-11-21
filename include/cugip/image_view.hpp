#pragma once

#include <cugip/detail/include.hpp>
#include <boost/mpl/bool.hpp>
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>
#include <cugip/image_locator.hpp>

namespace cugip {

template<typename TElement, size_t tDim = 2>
class device_image_view
{
public:
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
	operator[](coord_t aCoords)
	{
		return mData[aCoords];
	}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates)
	{
		return image_locator<this_t, TBorderHandling>(*this, aCoordinates);
	}

	CUGIP_DECL_HYBRID const memory_t&
	data() const
	{ return mData; }

protected:
	memory_t mData;

};

template<typename TElement, size_t tDim = 2>
class const_device_image_view
{
public:
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

	CUGIP_DECL_HYBRID const_value_type &
	operator[](coord_t aCoords)
	{
		return mData[aCoords];
	}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates)
	{
		return image_locator<this_t, TBorderHandling>(*this, aCoordinates);
	}

	CUGIP_DECL_HYBRID const memory_t&
	data() const
	{ return mData; }
protected:
	memory_t mData;
};

/** \ingroup  traits
 * @{
 **/
template<typename TView>
struct is_device_view: public boost::mpl::false_
{
	/*typedef boost::mpl::false_ type;
	static const bool value = type::value;*/
};

template<typename TElement, size_t tDim>
struct is_device_view<device_image_view<TElement, tDim> > : public boost::mpl::true_
{
	/*typedef boost::mpl::true_ type;
	static const bool value = type::value;*/
};

template<typename TElement, size_t tDim>
struct is_device_view<const_device_image_view<TElement, tDim> > : public boost::mpl::true_
{
	/*typedef boost::mpl::true_ type;
	static const bool value = type::value;*/
};


template<typename TElement, size_t tDim>
struct dimension<device_image_view<TElement, tDim> >: dimension_helper<tDim> {};

template<typename TElement, size_t tDim>
struct dimension<const_device_image_view<TElement, tDim> >: dimension_helper<tDim> {};

/**
 * @}
 **/

}//namespace cugip
