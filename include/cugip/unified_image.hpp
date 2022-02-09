#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/image_view.hpp>
#include <cugip/utils.hpp>
#include <cugip/cuda_utils.hpp>
#include <cugip/unified_image_view.hpp>
#include <memory>

namespace cugip {

//**************************************************************************
//Forward declarations
// template <typename TImage>
// typename TImage::view_t
// view(TImage &aImage);
//
// template <typename TImage>
// typename TImage::const_view_t
// const_view(TImage &aImage);
//**************************************************************************

template<typename TElement, int tDim>
class unified_image {
public:
	typedef unified_image_view<TElement, tDim> view_t;
	typedef const_unified_image_view<TElement, tDim> const_view_t;
	typedef TElement element_t;
	typedef TElement value_type;

	typedef typename dim_traits<tDim>::extents_t extents_t;


	unified_image() :
		mUnifiedPtr(nullptr, [](value_type *buffer) { cudaFree(buffer); })
	{}

	explicit unified_image(extents_t aExtents):
		mUnifiedPtr(nullptr, [](value_type *buffer) { cudaFree(buffer); })
	{
                reallocate(aExtents);
        }

	unified_image(int aS0, int aS1)
	{
                static_assert(tDim == 2, "Only 2-dimensional images can be specified by 2-dimensional extents!");
                reallocate(extents_t(aS0, aS1));
        }

	unified_image(int aS0, int aS1, int aS2)
	{
                static_assert(tDim == 3, "Only 3-dimensional images can be specified by 3-dimensional extents!");
                reallocate(extents_t(aS0, aS1, aS2));
        }

	unified_image(unified_image &&aOther) = default;
	unified_image & operator=(unified_image &&aOther) = default;

	unified_image & operator=(const unified_image &) = delete;
	unified_image(const unified_image &) = delete;

	extents_t
	dimensions() const
	{ return mSize; }


	value_type *
	pointer() const
	{
		return mUnifiedPtr.get();
	}

	extents_t
	strides() const
	{
		return mStrides;
	}

	void
	resize(extents_t aExtents)
	{
		reallocate(aExtents);
	}

	view_t
	view()
	{
		return view_t(mUnifiedPtr.get(), mSize, mStrides);
	}

	const_view_t
	const_view() const
	{
		return const_view_t(mUnifiedPtr.get(), mSize, mStrides);
	}

protected:
	void reallocate(const extents_t &aSize)
        {
		mUnifiedPtr.reset();
		value_type *ptr = nullptr;
		CUGIP_CHECK_RESULT(cudaMallocManaged(&ptr, sizeof(value_type) * product(coord_cast<int64_t>(aSize))));
                mUnifiedPtr = deleted_unique_ptr(ptr, [](value_type *buffer) { cudaFree(buffer); });
		mSize = aSize;
		mStrides = sizeof(value_type) * stridesFromSize(mSize);
        }

        extents_t mSize;
	extents_t mStrides;
	using deleted_unique_ptr = std::unique_ptr<value_type, void(*)(value_type*)>;
	deleted_unique_ptr mUnifiedPtr;

};

//**************************************************************************
//Image view construction
template<typename TElement, int tDim>
typename unified_image<TElement, tDim>::view_t
view(unified_image<TElement, tDim> &aImage)
{
	return aImage.view();
}

template<typename TElement, int tDim>
typename unified_image<TElement, tDim>::const_view_t
const_view(const unified_image<TElement, tDim> &aImage)
{
	return aImage.const_view();
}

/** \ingroup  traits
 * @{
 **/
template<typename TElement, int tDim>
struct dimension<unified_image<TElement, tDim> >: dimension_helper<tDim> {};

/**
 * @}
 **/

}//namespace cugip
