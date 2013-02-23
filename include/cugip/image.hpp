#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/image_view.hpp>
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>

namespace cugip {

//**************************************************************************
//Forward declarations
template <typename TImage>
typename TImage::view_t
view(TImage &aImage);

template <typename TImage>
typename TImage::const_view_t
const_view(TImage &aImage);
//**************************************************************************


template<typename TElement, size_t tDim = 2>
class device_image
{
public:
	typedef device_image_view<TElement, tDim> view_t;
	typedef const_device_image_view<TElement, tDim> const_view_t;
	typedef TElement element_t;
	typedef TElement value_type;

	typedef typename dim_traits<tDim>::extents_t extents_t;

	friend view_t view<>(device_image<TElement, tDim> &);
	friend const_view_t const_view<>(device_image<TElement, tDim> &);
public:
	device_image() 
	{}

	device_image(extents_t aExtents)
		: mData(aExtents)
	{}

	device_image(size_t aS0, size_t aS1 = 1, size_t aS2 = 1)
		: mData(typename dim_traits<tDim>::extents_t(aS0, aS1, aS2))
	{}

	CUGIP_DECL_HYBRID extents_t 
	dimensions() const
	{ return mData.dimensions(); }
protected:
	device_image & operator=(const device_image &);
	device_image(const device_image &);

	typename memory_management<TElement, tDim>::device_memory_owner mData;
	//device_ptr<element_t> mData;
};

//**************************************************************************
//Image view construction
template <typename TImage>
typename TImage::view_t
view(TImage &aImage)
{
	return typename TImage::view_t(aImage.mData);
}

template <typename TImage>
typename TImage::const_view_t
const_view(TImage &aImage)
{
	return typename TImage::const_view_t(aImage.mData);
}



}//namespace cugip
