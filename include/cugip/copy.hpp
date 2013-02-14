#pragma once

#include <cugip/detail/include.hpp>

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


