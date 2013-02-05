#pragma once

#include "ImageView.hpp"
#include "utils.hpp"

namespace cugip {

template<typename TElement>
class device_image
{
public:
	device_image() 
	{}

	device_image(size_t aWidth, size_t aHeight)
	{}
protected:
	device_ptr<TElement> mData;
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
