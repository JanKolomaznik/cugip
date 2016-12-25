#pragma once

#include <type_traits>

#include <cugip/cuda_utils.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/device_flag.hpp>


namespace cugip {

/** \addtogroup math
 * @{
 **/

struct DummyGlobalState
{
	void
	initialize() {}

	template<typename TView>
	void
	preprocess(TView aView) {}

	template<typename TView>
	void
	postprocess(TView aView) {}
};

template<typename TBaseClass>
struct DeviceFlagMixin : TBaseClass
{
	// TODO fix initialization from initializer list
	void
	initialize(){
		TBaseClass::initialize();
		mDeviceFlag.reset_host();
	}

	template<typename TView>
	void
	preprocess(TView aView)
	{
		mDeviceFlag.reset_host();
	}

	CUGIP_DECL_DEVICE
	void
	signal()
	{
		mDeviceFlag.set_device();
	}

	bool
	is_finished()
	{
		return !mDeviceFlag.check_host();
	}


	device_flag_view mDeviceFlag;
};

typedef DeviceFlagMixin<DummyGlobalState> ConvergenceFlag;


/**
 * @}
 **/

} // namespace cugip
