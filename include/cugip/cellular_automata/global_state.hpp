#pragma once

#include <type_traits>

#include <cugip/cuda_utils.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/device_flag.hpp>


namespace cugip {


struct DummyGlobalState
{
	void
	initialize() {}

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

	CUGIP_DECL_DEVICE
	void
	signal()
	{
		mDeviceFlag.set_device();
	}

	device_flag_view mDeviceFlag;
};

typedef DeviceFlagMixin<DummyGlobalState> ConvergenceFlag;

} // namespace cugip
