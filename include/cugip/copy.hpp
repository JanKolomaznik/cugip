#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/memory_view.hpp>

namespace cugip {

namespace detail {

template<typename TType>
void
copy_wrapper(
	const TType *aFromPointer,
	size_t aFromPitch,
	TType *aToPointer,
	size_t aToPitch,
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
	size_t aFromPitch,
	TType *aToPointer,
	size_t aToPitch,
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
			% ((size_t)aFrom.data().mData.p)
			% ((size_t)aTo.data().mData.p)
			));
		/*copy_wrapper(aFrom.data().mData.p,
			      aFrom.data().mPitch,
				aTo.data().mData.p,
			      aTo.data().mPitch,
				aTo.dimensions(),
				cudaMemcpyDeviceToDevice);*/
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
			% ((size_t) src)
			% ((size_t)aTo.data().mData.p)
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
			% ((size_t)aFrom.data().mData.p)
			% ((size_t) dst)
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


}//namespace cugip


