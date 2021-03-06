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


template<typename TElement, int tDim = 2>
class device_image
{
public:
	typedef device_image_view<TElement, tDim> view_t;
	typedef const_device_image_view<TElement, tDim> const_view_t;
	typedef TElement element_t;
	typedef TElement value_type;

	typedef typename dim_traits<tDim>::extents_t extents_t;

	//friend view_t view<>(device_image<TElement, tDim> &);
	//friend const_view_t const_view<>(device_image<TElement, tDim> &);
public:
	device_image()
	{}

	device_image(extents_t aExtents)
		: mData(aExtents)
	{}

	device_image(int aS0)
		: mData(typename dim_traits<tDim>::extents_t(aS0))
	{}

	device_image(int aS0, int aS1)
		: mData(typename dim_traits<tDim>::extents_t(aS0, aS1))
	{}

	device_image(int aS0, int aS1, int aS2)
		: mData(typename dim_traits<tDim>::extents_t(aS0, aS1, aS2))
	{}

	device_image(device_image &&aOther) = default;
	device_image & operator=(device_image &&aOther) = default;

	device_image & operator=(const device_image &) = delete;
	device_image(const device_image &) = delete;

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mData.dimensions(); }

	view_t
	view()
	{
		//return view_t(mData);
		return view_t(pointer(), dimensions(), strides());
	}

	const_view_t
	const_view() const
	{
		//return const_view_t(mData);
		return const_view_t(pointer(), dimensions(), strides());
	}

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


	void
	resize(extents_t aExtents)
	{
		mData.reallocate(aExtents);
	}

	void
	reset()
	{
		mData.reset();
	}

protected:

	typename memory_management<TElement, tDim>::device_memory_owner mData;
	//device_ptr<element_t> mData;
};

//**************************************************************************
//Image view construction
template <typename TImage>
typename TImage::view_t
view(TImage &aImage)
{
	return aImage.view();
}

template <typename TImage>
typename TImage::const_view_t
const_view(TImage &aImage)
{
	return aImage.const_view();
}

/** \ingroup  traits
 * @{
 **/
template<typename TElement, int tDim>
struct dimension<device_image<TElement, tDim> >: dimension_helper<tDim> {};

/**
 * @}
 **/

}//namespace cugip
