#pragma once

namespace cugip {

namespace detail {

	template<bool tFromDevice, bool tToDevice>
	struct copy_methods;

	//device to device
	template<true, true>
	struct copy_methods
	{
		template<typename TFrom, typename TTo>
		static void
		copy(TFrom &aFrom, TTo &aTo)
		{

		}
	};

	//host to device
	template<false, true>
	struct copy_methods
	{
		template<typename TFrom, typename TTo>
		static void
		copy(TFrom &aFrom, TTo &aTo)
		{

		}
	};

	//host to host
	template<false, false>
	struct copy_methods
	{
		template<typename TFrom, typename TTo>
		static void
		copy(TFrom &aFrom, TTo &aTo)
		{

		}
	};

	//device to host
	template<true, false>
	struct copy_methods
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
	copy_methods<is_device_view<TFrom>::value, is_device_view<TTo>::value>::
		copy(aFrom, aTo);
}


}//namespace cugip


