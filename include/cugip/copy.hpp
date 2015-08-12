#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>
#include <cugip/cuda_utils.hpp>
#include <cugip/math.hpp>
#include <cugip/memory_view.hpp>

namespace cugip {
/*
namespace detail {

template<typename TType>
void
copy_wrapper(
	const TType *aFromPointer,
	int aFromPitch,
	TType *aToPointer,
	int aToPitch,
	size2_t aSize,
	cudaMemcpyKind aMemcpyKind)
{
	CUGIP_CHECK_RESULT(cudaMemcpy2D(
			aToPointer,
			aToPitch,
			aFromPointer,
			aFromPitch,
			aSize[0]*sizeof(TType),
			aSize[1],
			aMemcpyKind));
}

template<typename TType>
void
copy_wrapper(
	const TType *aFromPointer,
	int aFromPitch,
	TType *aToPointer,
	int aToPitch,
	size3_t aSize,
	cudaMemcpyKind aMemcpyKind)
{
	CUGIP_CHECK_RESULT(cudaMemcpy2D(
			aToPointer,
			aToPitch,
			aFromPointer,
			aFromPitch,
			aSize[0]*sizeof(TType),
			aSize[1] * aSize[2],
			aMemcpyKind));
}

template<bool tFromDevice, bool tToDevice>
struct copy_methods_impl;

//device to device
template<>
struct copy_methods_impl<true, true>
{
	template<typename TFrom, typename TTo>
	static void
	copy(TFrom &aFrom, TTo &aTo)
	{
		D_PRINT("COPY: device to device");

		CUGIP_ASSERT(aTo.dimensions() == aFrom.dimensions());

		unsigned char *dst = reinterpret_cast<unsigned char*>(&(aTo.pixels()(0,0)));
		int diff = reinterpret_cast<unsigned char*>(&(aTo.pixels()(0,1))) - dst;
		CUGIP_ASSERT(diff >= 0);

		D_PRINT(boost::str(boost::format("COPY: device to device, %1$#x => %2$#x")
			% ((int)aFrom.data().mData.p)
			% ((int)aTo.data().mData.p)
			));
		//copy_wrapper(aFrom.data().mData.p,
		//	      aFrom.data().mPitch,
		//		aTo.data().mData.p,
		//	      aTo.data().mPitch,
		//		aTo.dimensions(),
		//		cudaMemcpyDeviceToDevice);
		CUGIP_CHECK_RESULT(cudaMemcpy2D(aTo.data().mData.p,
			      aTo.data().mPitch,
			      aFrom.data().mData.p,
			      aFrom.data().mPitch,
			      get<0>(aTo.dimensions())*sizeof(typename TTo::value_type),
			      get<1>(aTo.dimensions()),
			      cudaMemcpyDeviceToDevice));
	}
};

//host to device
template<>
struct copy_methods_impl<false, true>
{
	template<typename TFrom, typename TTo>
	static void
	copy(TFrom &aFrom, TTo &aTo)
	{
		D_PRINT("COPY: host to device");

		CUGIP_ASSERT(aFrom.width() == aTo.dimensions().template get<0>());
		CUGIP_ASSERT(aFrom.height() == aTo.dimensions().template get<1>());


		const unsigned char *src = reinterpret_cast<const unsigned char*>(&(aFrom.pixels()(0,0)));
		int diff = reinterpret_cast<const unsigned char*>(&(aFrom.pixels()(0,1))) - src;
		CUGIP_ASSERT(diff >= 0);

		D_PRINT(boost::str(boost::format("COPY: host to device, %1$#x => %2$#x")
			% ((int) src)
			% ((int)aTo.data().mData.p)
			));

		CUGIP_CHECK_RESULT(cudaMemcpy2D(aTo.data().mData.p,
			      aTo.data().mPitch,
			      src,
			      diff,
			      aFrom.width()*sizeof(typename TFrom::value_type),
			      aFrom.height(),
			      cudaMemcpyHostToDevice));
		cudaThreadSynchronize();
	}
};

//host to host
template<>
struct copy_methods_impl<false, false>
{
	template<typename TFrom, typename TTo>
	static void
	copy(TFrom &aFrom, TTo &aTo)
	{
		D_PRINT("COPY: host to host");

		CUGIP_ASSERT(false && "Not implemented");
	}
};

//device to host
template<>
struct copy_methods_impl<true, false>
{
	template<typename TFrom, typename TTo>
	static void
	copy(TFrom &aFrom, TTo &aTo)
	{
		CUGIP_ASSERT(aTo.width() == aFrom.dimensions().template get<0>());
		CUGIP_ASSERT(aTo.height() == aFrom.dimensions().template get<1>());

		unsigned char *dst = reinterpret_cast<unsigned char*>(&(aTo.pixels()(0,0)));
		int diff = reinterpret_cast<unsigned char*>(&(aTo.pixels()(0,1))) - dst;
		CUGIP_ASSERT(diff >= 0);

		D_PRINT(boost::str(boost::format("COPY: device to host, %1$#x => %2$#x")
			% ((int)aFrom.data().mData.p)
			% ((int) dst)
			));
		CUGIP_CHECK_RESULT(cudaMemcpy2D(dst,
			      diff,
			      aFrom.data().mData.p,
			      aFrom.data().mPitch,
			      aTo.width()*sizeof(typename TTo::value_type),
			      aTo.height(),
			      cudaMemcpyDeviceToHost));
	}
};

}//namespace detail


template<typename TFrom, typename TTo>
void
copy(TFrom aFrom, TTo aTo)
{
	cugip::detail::copy_methods_impl<
			cugip::is_device_view<TFrom>::value,
			cugip::is_device_view<TTo>::value,
		>::copy(aFrom, aTo);
}

template<typename TFrom, typename TTo>
void
copy_to(TFrom aFrom, TTo aTo)
{
	detail::copy_wrapper(
		aFrom.data().mData,
		aFrom.data().mPitch,
		aTo.data().mData.p,
		aTo.data().mPitch,
		aFrom.dimensions(),
		cudaMemcpyHostToDevice);
}

template<typename TFrom, typename TTo>
void
copy_from(TFrom aFrom, TTo aTo)
{
	detail::copy_wrapper(
		aFrom.data().mData.p,
		aFrom.data().mPitch,
		aTo.data().mData,
		aTo.data().mPitch,
		aFrom.dimensions(),
		cudaMemcpyDeviceToHost);
}
*/
template<bool tFromDevice, bool tToDevice>
struct CopyDirectionTag {};

typedef CopyDirectionTag<true, true> DeviceToDeviceTag;
typedef CopyDirectionTag<true, false> DeviceToHostTag;
typedef CopyDirectionTag<false, false> HostToHostTag;
typedef CopyDirectionTag<false, true> HostToDeviceTag;

/// Asynchronous copy between compatible image views.
/// Device/host direction is defined by the type of these views.
/// Copying views between host <-> device must be done through memory based views.
/// \param from_view Source
/// \param to_view Target
/// \param cuda_stream Selected CUDA stream.
template <typename TFromView, typename TToView>
void copyAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream = 0);


/// Synchronous copy between compatible image views.
/// \sa CopyAsync()
template <typename TFromView, typename TToView>
void copy(
	TFromView from_view,
	TToView to_view);

// IMPLEMENTATION - TODO(reorganize)

template <typename TFromView, typename TToView>
CUGIP_GLOBAL void copyKernel(
	TFromView from_view,
	TToView to_view)
{
	int element_count = elementCount(from_view);
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + tid;
	int grid_size = blockDim.x * gridDim.x;

	while (index < element_count) {
		linear_access(to_view, index) = linear_access(from_view, index);
		index += grid_size;
	}
	__syncthreads();
}


template <typename TFromView, typename TToView>
void copyDeviceToDeviceAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	// TODO(johny) - use memcpy for memory based views
	constexpr int cBucketSize = 4;  // Bundle more computation in one block

	dim3 block(512, 1, 1);
	dim3 grid(1 + (elementCount(from_view) - 1) / (block.x * cBucketSize), 1, 1);

	copyKernel<TFromView, TToView><<<grid, block, 0, cuda_stream>>>(from_view, to_view);
	CUGIP_CHECK_ERROR_STATE("After CopyKernel");
}


template <typename TFromView, typename TToView>
void copyDeviceToHostAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	typedef typename std::remove_cv<typename TFromView::value_type>::type FromElement;
	typedef typename std::remove_cv<typename TToView::value_type>::type ToElement;
	static_assert(std::is_same<FromElement, ToElement>::value, "From/To views have incompatible element types.");
	//static_assert(TFromView::kIsMemoryBased, "Source view must be memory based");
	//static_assert(TToView::kIsMemoryBased, "Target view must be memory based");
	// Copy without padding
	cudaMemcpy3DParms parameters = { 0 };

	parameters.srcPtr = stridesToPitchedPtr(from_view.pointer(), from_view.dimensions(), from_view.strides());
	parameters.dstPtr = stridesToPitchedPtr(to_view.pointer(), to_view.dimensions(), to_view.strides());
	parameters.extent = makeCudaExtent(sizeof(typename TFromView::value_type), from_view.dimensions());
	parameters.kind = cudaMemcpyDeviceToHost;
	CUGIP_DFORMAT("Copy pitched data: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcPtr, parameters.dstPtr, parameters.extent);
	CUGIP_CHECK_RESULT(cudaMemcpy3DAsync(&parameters, cuda_stream));
}


template <typename TFromView, typename TToView>
void copyHostToDeviceAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	typedef typename std::remove_cv<typename TFromView::value_type>::type FromElement;
	typedef typename std::remove_cv<typename TToView::value_type>::type ToElement;
	static_assert(std::is_same<FromElement, ToElement>::value, "From/To views have incompatible element types.");
	//static_assert(TFromView::kIsMemoryBased, "Source view must be memory based");
	//static_assert(TToView::kIsMemoryBased, "Target view must be memory based");
	cudaMemcpy3DParms parameters = { 0 };

	parameters.srcPtr = stridesToPitchedPtr(from_view.pointer(), from_view.dimensions(), from_view.strides());
	parameters.dstPtr = stridesToPitchedPtr(to_view.pointer(), to_view.dimensions(), to_view.strides());
	parameters.extent = makeCudaExtent(sizeof(typename TFromView::value_type), from_view.dimensions());
	parameters.kind = cudaMemcpyHostToDevice;

	CUGIP_DFORMAT("Copy pitched data: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcPtr, parameters.dstPtr, parameters.extent);
	CUGIP_CHECK_RESULT(cudaMemcpy3DAsync(&parameters, cuda_stream));
}


template <typename TFromView, typename TToView>
void asyncCopyHelper(
	TFromView from_view,
	TToView to_view,
	DeviceToDeviceTag /*tag*/,
	cudaStream_t cuda_stream)
{
	copyDeviceToDeviceAsync(from_view, to_view, cuda_stream);
}


template <typename TFromView, typename TToView>
void asyncCopyHelper(
	TFromView from_view,
	TToView to_view,
	DeviceToHostTag /*tag*/,
	cudaStream_t cuda_stream)
{
	copyDeviceToHostAsync(from_view, to_view, cuda_stream);
}


template <typename TFromView, typename TToView>
void asyncCopyHelper(
	TFromView from_view,
	TToView to_view,
	HostToDeviceTag /*tag*/,
	cudaStream_t cuda_stream)
{
	copyHostToDeviceAsync(from_view, to_view, cuda_stream);
}


// TODO(johny) implement special cases, unified memory, etc.

template <typename TFromView, typename TToView>
void copyAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	CUGIP_DFORMAT("Copy sizes: \n  src: %1%\n  dst: %2%", from_view.dimensions(), to_view.dimensions());
	if (from_view.dimensions() != to_view.dimensions()) {
		CUGIP_THROW(IncompatibleViewSizes() /*<< GetViewPairSizesErrorInfo(from_view.dimensions(), to_view.dimensions()))*/);
	}

	static_assert(is_device_view<TFromView>::value || is_device_view<TToView>::value, "Host to host copy not yet implemented - decide sycnhronous/asynchronous behavior");
	asyncCopyHelper(from_view, to_view, CopyDirectionTag<is_device_view<TFromView>::value, is_device_view<TToView>::value>(), cuda_stream);
}


template <typename TFromView, typename TToView>
void copy(
	TFromView from_view,
	TToView to_view)
{
	copyAsync(from_view, to_view);
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
}


}//namespace cugip
