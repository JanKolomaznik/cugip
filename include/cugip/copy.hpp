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


