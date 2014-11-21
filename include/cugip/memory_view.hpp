#pragma once

#include <cugip/detail/include.hpp>
#include <boost/mpl/bool.hpp>
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>
#include <cugip/image_locator.hpp>

namespace cugip {

template<typename TElement, size_t tDim = 2>
class memory_view
{
public:
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	typedef typename memory_management<TElement, tDim>::host_memory memory_t;
	typedef memory_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef value_type & accessed_type;

	memory_view(const memory_t &aData) :
		mData(aData)
	{}

	memory_view()
	{ /*empty*/ }

	extents_t
	dimensions() const
	{ return mData.dimensions(); }

	accessed_type
	operator[](coord_t aCoords) const
	{
		return mData[aCoords];
	}

	/*template<typename TBorderHandling>
	image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates)
	{
		return image_locator<this_t, TBorderHandling>(*this, aCoordinates);
	}*/

	const memory_t&
	data() const
	{ return mData; }

protected:
	memory_t mData;

};

template<typename TElement, size_t tDim = 2>
class const_memory_view
{
public:
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename dim_traits<tDim>::diff_t diff_t;
	typedef typename memory_management<TElement, tDim>::const_host_memory memory_t;
	typedef const_memory_view<TElement, tDim> this_t;
	typedef TElement value_type;
	typedef const TElement const_value_type;
	typedef const_value_type & accessed_type;

	const_memory_view(const memory_t &aData) :
		mData(aData)
	{}

	const_memory_view(const typename memory_management<TElement, tDim>::host_memory &aData) :
		mData(aData)
	{}

	const_memory_view()
	{ /*empty*/ }

	extents_t
	dimensions() const
	{ return mData.dimensions(); }

	const_value_type &
	operator[](coord_t aCoords)
	{
		return mData[aCoords];
	}

	/*template<typename TBorderHandling>
	image_locator<this_t, TBorderHandling>
	locator(coord_t aCoordinates)
	{
		return image_locator<this_t, TBorderHandling>(*this, aCoordinates);
	}*/

	const memory_t&
	data() const
	{ return mData; }
protected:
	memory_t mData;
};

/** \ingroup  traits
 * @{
 **/

template<typename TElement, size_t tDim>
struct dimension<memory_view<TElement, tDim> >: dimension_helper<tDim> {};

template<typename TElement, size_t tDim>
struct dimension<const_memory_view<TElement, tDim> >: dimension_helper<tDim> {};

template<typename TView>
struct is_memory_view: public boost::mpl::false_ {};

template<typename TElement, size_t tDim>
struct is_memory_view<memory_view<TElement, tDim> > : public boost::mpl::true_ {};

template<typename TElement, size_t tDim>
struct is_memory_view<const_memory_view<TElement, tDim> > : public boost::mpl::true_ {};



/**
 * @}
 **/

}//namespace cugip

