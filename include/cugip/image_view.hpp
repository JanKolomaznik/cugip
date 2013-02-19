#pragma once

#include <cugip/detail/include.hpp>
#include <boost/mpl/bool.hpp>
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>

namespace cugip {

template<typename TElement, size_t tDim = 2>
class device_image_view
{
public:
	typedef typename dim_traits<tDim>::extents_t extents_t;
	typedef typename dim_traits<tDim>::coord_t coord_t;
	typedef typename memory_management<TElement, tDim>::device_memory memory_t;
	typedef TElement value_type;

	device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}

	device_image_view() 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID extents_t 
	size() const
	{ return mData.size(); }

	CUGIL_DECL_HYBRID value_type &
	operator[](coord_t aCoords)
	{
		return mData[aCoords];
	}


	CUGIL_DECL_HYBRID const memory_t&
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
	typedef typename memory_management<TElement, tDim>::const_device_memory memory_t;
	typedef TElement value_type;

	const_device_image_view(const typename memory_management<TElement, tDim>::const_device_memory &aData) :
		mData(aData)
	{}

	const_device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}

	const_device_image_view() 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID extents_t 
	size() const
	{ return mData.size(); }

	CUGIL_DECL_HYBRID const memory_t&
	data() const
	{ return mData; }
protected:
	memory_t mData;
};

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

}//namespace cugip
