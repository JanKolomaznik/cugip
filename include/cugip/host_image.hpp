#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/image_view.hpp>
#include <cugip/utils.hpp>
//#include <cugip/cuda_utils.hpp>
#include <cugip/host_image_view.hpp>

namespace cugip {

//**************************************************************************
//Forward declarations
template <typename TImage>
typename TImage::view_t
view(const TImage &aImage);

template <typename TImage>
typename TImage::const_view_t
const_view(const TImage &aImage);
//**************************************************************************

/// \return Strides for memory without padding.
/*CUGIP_DECL_HYBRID
inline Int2 stridesFromExtents(Int2 size) {
	return Int2(1, size[0]);
}

/// \return Strides for memory without padding.
CUGIP_DECL_HYBRID
inline Int3 stridesFromExtents(Int3 size) {
	return Int3(1, size[0], size[0] * size[1]);
}*/


template<typename TElement, int tDim = 2>
class host_image
{
public:
	typedef host_image_view<TElement, tDim> view_t;
	typedef const_host_image_view<TElement, tDim> const_view_t;
	typedef TElement element_t;
	typedef TElement value_type;

	typedef typename dim_traits<tDim>::extents_t extents_t;

	host_image()
	{}

	host_image(extents_t aExtents)
	{
                reallocate(aExtents);
        }

	host_image(int aS0, int aS1)
	{
                static_assert(tDim == 2, "Only 2-dimensional images can be specified by 2-dimensional extents!");
                reallocate(extents_t(aS0, aS1));
        }

	host_image(int aS0, int aS1, int aS2)
	{
                static_assert(tDim == 3, "Only 3-dimensional images can be specified by 3-dimensional extents!");
                reallocate(extents_t(aS0, aS1, aS2));
        }

	host_image(host_image &&aOther) = default;
	host_image & operator=(host_image &&aOther) = default;

	host_image & operator=(const host_image &) = delete;
	host_image(const host_image &) = delete;

	extents_t
	dimensions() const
	{ return mSize; }

	view_t
	view() const
	{
		return view_t(mHostPtr.get(), mSize, mStrides);
	}

	const_view_t
	const_view() const
	{
		return const_view_t(mHostPtr.get(), mSize, mStrides);
	}

	value_type *
	pointer() const
	{
		return mHostPtr.get();
	}

	extents_t
	strides() const
	{
		return mStrides;
	}
protected:
        void reallocate(const extents_t &aSize)
        {
                mHostPtr.reset(new value_type[product(coord_cast<int64_t>(aSize))]);
		mSize = aSize;
		mStrides = sizeof(value_type) * stridesFromSize(mSize);
        }

        extents_t mSize;
	extents_t mStrides;
	std::unique_ptr<value_type []> mHostPtr;
};

//**************************************************************************
//Image view construction
template<typename TElement, int tDim>
typename host_image<TElement, tDim>::view_t
view(const host_image<TElement, tDim> &aImage)
{
	return aImage.view();
}

template<typename TElement, int tDim>
typename host_image<TElement, tDim>::const_view_t
const_view(const host_image<TElement, tDim> &aImage)
{
	return aImage.const_view();
}

/** \ingroup  traits
 * @{
 **/
template<typename TElement, int tDim>
struct dimension<host_image<TElement, tDim> >: dimension_helper<tDim> {};

/**
 * @}
 **/

}//namespace cugip
