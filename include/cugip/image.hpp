#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/image_view.hpp>
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>

namespace cugip {

template<typename TElement, size_t tDim = 2>
class device_image
{
public:
	typedef device_image_view<TElement, tDim> view_t;
	typedef const_device_image_view<TElement, tDim> const_view_t;
	typedef TElement element_t;

	typedef typename dim_traits<tDim>::extents_t extents_t;
public:
	device_image() 
	{}

	device_image(extents_t aExtents)
	{}

	device_image(size_t aS0, size_t aS1 = 1, size_t aS2 = 1)
		: mData(typename dim_traits<tDim>::extents_t(aS0, aS1, aS2))
	{}
protected:
	typename memory_management<TElement, tDim>::device_memory_owner mData;
	//device_ptr<element_t> mData;
};

template <typename TImage>
typename TImage::view_t
view(TImage &aImage)
{
	return typename TImage::view_t();
}

template <typename TImage>
typename TImage::const_view_t
const_view(TImage &aImage)
{
	return typename TImage::const_view_t();
}



}//namespace cugip
