#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>

namespace cugip {

namespace detail {

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

			CUGIL_ASSERT(false && "Not implemented");
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
			CUGIL_ASSERT(aFrom.width() == aTo.size().template get<0>());
			CUGIL_ASSERT(aFrom.height() == aTo.size().template get<1>());


			const char *src = reinterpret_cast<const char*>(&(aFrom.pixels()(0,0)));
			int diff = reinterpret_cast<const char*>(&(aFrom.pixels()(0,1))) - src;
			CUGIL_ASSERT(diff >= 0);

			cudaMemcpy2D (aTo.data().mData.p, 
				      aTo.data().mPitch, 
				      src, 
				      diff, 
				      aFrom.width()*sizeof(typename TFrom::value_type), 
				      aFrom.height(), 
				      cudaMemcpyHostToDevice);
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

			CUGIL_ASSERT(false && "Not implemented");
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
			D_PRINT("COPY: device to host");
			CUGIL_ASSERT(aTo.width() == aFrom.size().template get<0>());
			CUGIL_ASSERT(aTo.height() == aFrom.size().template get<1>());

			char *dst = reinterpret_cast<char*>(&(aTo.pixels()(0,0)));
			int diff = reinterpret_cast<char*>(&(aTo.pixels()(0,1))) - dst;
			CUGIL_ASSERT(diff >= 0);

			cudaMemcpy2D (dst, 
				      diff,
				      aFrom.data().mData.p, 
				      aFrom.data().mPitch,
				      aTo.width()*sizeof(typename TTo::value_type), 
				      aTo.height(), 
				      cudaMemcpyDeviceToHost);

		}
	};

}//namespace detail


template<typename TFrom, typename TTo>
void
copy(TFrom aFrom, TTo aTo)
{
	cugip::detail::copy_methods_impl<
			cugip::is_device_view<TFrom>::value, 
			cugip::is_device_view<TTo>::value
		>::copy(aFrom, aTo);
}


}//namespace cugip


