#pragma once
#include <cugip/utils.hpp>
#include <cugip/memory.hpp>

namespace cugip {

struct device_flag_view
{
	/*typedef bool flag_t;
	CUGIP_DECL_HYBRID void
	reset()
	{
		mFlag.assign(true);
		// *mFlag = false;
	}*/

	typedef bool flag_t;
	CUGIP_DECL_HOST void
	reset_host()
	{
		mFlag.assign_host(false);
		//*mFlag = false;
	}

	CUGIP_DECL_HOST void
	set_host()
	{
		mFlag.assign_host(true);
		//*mFlag = true;
	}

	CUGIP_DECL_DEVICE void
	set_device()
	{
		mFlag.assign_device(true);
		//*mFlag = true;
	}

	CUGIP_DECL_HOST bool
	check_host()
	{
		return mFlag.retrieve_host();
	}


	/*CUGIP_DECL_HYBRID
	operator bool() const
	{
		bool value;// = *mFlag;

#ifndef __CUDACC__
		D_PRINT(boost::str(boost::format("Checking device flag ... %1%") % value));
#endif
		return value;
	}*/


	device_ptr<flag_t> mFlag;
};

struct device_flag: public device_flag_view
{
	typedef bool flag_t;
	device_flag()
	{
		void *devPtr = NULL;
		CUGIP_CHECK_RESULT(cudaMalloc(&devPtr, sizeof(flag_t)));
		mFlag = reinterpret_cast<flag_t *>(devPtr);
		reset_host();
	}

	~device_flag()
	{
		CUGIP_ASSERT(mFlag);
		cudaFree(mFlag.get());
	}

	device_flag_view &
	view()
	{ return static_cast<device_flag_view &>(*this); }
private:
	// non-copyable
	device_flag(const device_flag &);
	device_flag &
	operator=(const device_flag &);
};



}//namespace cugip
