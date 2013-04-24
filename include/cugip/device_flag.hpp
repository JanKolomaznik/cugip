#pragma once
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>

namespace cugip {

struct device_flag_view
{
	CUGIP_DECL_HYBRID void
	reset()
	{
		*mFlag = false;
	}

	CUGIP_DECL_HYBRID void
	set()
	{
		*mFlag = true;
	}

	device_ptr<flag_t> mFlag;
};

struct device_flag: public device_flag_view
{
	typedef bool flag_t;
	device_flag() 
	{
		void *devPtr = NULL;
		CUGIP_CHECK_RESULT(cudaMalloc(&devPtr, sizeof(flag_t)));
		mFlag = reinterpret_cast<flag_t>(devPtr);
		reset();
	}

	~device_flag()
	{
		CUGIP_ASSERT(mFlag);
		cudaFree(mFlag.get());
	}

};



}//namespace cugip
