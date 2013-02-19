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

	device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}

	device_image_view() 
	{ /*empty*/ }

	extents_t size() const
	{ return mData.size(); }
protected:
	typename memory_management<TElement, tDim>::device_memory mData;

};

template<typename TElement, size_t tDim = 2>
class const_device_image_view
{
public:
	typedef typename dim_traits<tDim>::extents_t extents_t;

	const_device_image_view(const typename memory_management<TElement, tDim>::const_device_memory &aData) :
		mData(aData)
	{}

	const_device_image_view(const typename memory_management<TElement, tDim>::device_memory &aData) :
		mData(aData)
	{}

	const_device_image_view() 
	{ /*empty*/ }

	extents_t size() const
	{ return mData.size(); }
protected:
	typename memory_management<TElement, tDim>::const_device_memory mData;
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
