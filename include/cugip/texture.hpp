#pragma once

#include <cugip/math.hpp>
#include <cugip/functors.hpp>
#include <cugip/tuple.hpp>
#include <cugip/memory.hpp>
#include <cugip/detail/view_declaration_utils.hpp>

namespace cugip {


template<typename TElement, int tDim>
class texture_view
{
public:
	typedef TElement element_t;
	typedef TElement value_type;

	typedef typename dim_traits<tDim>::extents_t extents_t;

	texture_view(cudaTextureObject_t aTexture, extents_t aSize)
		: mTexture(aTexture)
		, mSize(aSize)
	{}


	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mSize; }

protected:
	cudaTextureObject_t mTexture;
	extents_t mSize;
};

CUGIP_DECLARE_DEVICE_VIEW_TRAITS((texture_view<TElement, tDim>), tDim, typename TElement, int tDim);

template<typename TElement, int tDim>
class texture_image
{
public:
	typedef texture_view<TElement, tDim> view_t;
	typedef texture_view<TElement, tDim> const_view_t;
	typedef TElement element_t;
	typedef TElement value_type;

	typedef typename dim_traits<tDim>::extents_t extents_t;


	texture_image(extents_t aDimensions)
		: mData(aDimensions)
	{}

	~texture_image()
	{
		finalize();
	}

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mData.dimensions(); }

	void
	allocate(extents_t aSize)
	{
		cudaResourceDesc resDesc = { 0 };
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = mData.mData.get();
		resDesc.res.linear.desc.f = cudaChannelFormatKindSigned; // TODO
		resDesc.res.linear.desc.x = 8 * sizeof(TElement); // bits per channel
		resDesc.res.linear.sizeInBytes = aSize * sizeof(TElement);

		cudaTextureDesc texDesc = { 0 };
		texDesc.readMode = cudaReadModeElementType;

		cudaCreateTextureObject(&mTexture, &resDesc, &texDesc, NULL);
	}

	void finalize()
	{
		cudaDestroyTextureObject(mTexture);
	}

	view_t
	view() const
	{
		return view_t(mTexture, mData.dimensions());
	}

	view_t
	const_view() const
	{
		return view();
	}

	typename memory_management<TElement, tDim>::device_memory_owner mData;
	cudaTextureObject_t mTexture;
};

template<typename TElement, int tDim>
typename texture_image<TElement, tDim>::view_t
view(texture_image<TElement, tDim> &aImage)
{
	return aImage.view();
}

template<typename TElement, int tDim>
typename texture_image<TElement, tDim>::view_t
const_view(const texture_image<TElement, tDim> &aImage)
{
	return aImage.view();
}

/** \ingroup  traits
 * @{
 **/
template<typename TElement, int tDim>
struct dimension<texture_image<TElement, tDim> >: dimension_helper<tDim> {};


template <typename TElement, int tDim, typename TToView>
void asyncCopyHelper(
	texture_view<TElement, tDim> from_view,
	TToView to_view,
	DeviceToHostTag /*tag*/,
	cudaStream_t cuda_stream)
{
	CUGIP_ASSERT(false);
	//copyDeviceToHostAsync(from_view, to_view, cuda_stream);
}


template <typename TFromView, typename TElement, int tDim>
void asyncCopyHelper(
	TFromView from_view,
	texture_view<TElement, tDim> to_view,
	HostToDeviceTag /*tag*/,
	cudaStream_t cuda_stream)
{
	CUGIP_ASSERT(false);
	//copyHostToDeviceAsync(from_view, to_view, cuda_stream);
}



} // namespace cugip
