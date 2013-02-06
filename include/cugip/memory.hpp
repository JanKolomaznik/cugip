#pragma once

#include "utils.hpp"

namespace cugip {

template<typename TType>
struct device_ptr
{
	CUGIL_DECL_HYBRID
	device_ptr(): p(0) 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID
	device_ptr(const device_ptr &aArg): p(aArg.p) 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID TType * 
	operator->()
	{ return p; }

	CUGIL_DECL_HYBRID device_ptr &
	operator=(const device_ptr &aArg)
	{ p = aArg.p; }

	CUGIL_DECL_HYBRID device_ptr &
	operator=(TType *aArg)
	{ p = aArg; }

	CUGIL_DECL_HYBRID
	operator bool() const
	{ return p != 0; }

	TType *p;
};

template<typename TType>
struct device_memory_1d
{
	
};

template<typename TType>
struct device_memory_2d
{
	device_memory_2d()
	{}

	device_memory_2d(device_ptr<TType> aPtr, size_t aWidth, size_t aHeight, size_t aPitch)
		:mData(aPtr), mWidth(aWidth), mHeight(aHeight), mPitch(aPitch)
	{
		CUGIL_ASSERT(mPitch >= (mWidth*sizeof(TType)));
	}

	~device_memory_2d()
	{ }

	device_ptr<TType> mData;
	size_t mWidth;
	size_t mHeight;
	size_t mPitch;
};



template<typename TType>
struct device_memory_2d_owner: public device_memory_2d<TType>
{
	device_memory_2d_owner()
	{}

	device_memory_2d_owner(size_t aWidth, size_t aHeight)
	{
		void *devPtr = NULL;
		CUGIL_CHECK_RESULT(cudaMallocPitch(&devPtr, &(this->mPitch), aWidth * sizeof(TType), aHeight));
		this->mWidth = aWidth;
		this->mHeight = aHeight;
		this->mData = reinterpret_cast<TType*>(devPtr);
	}

	~device_memory_2d()
	{
		if (this->mData) {
			CUGIL_ASSERT_RESULT(cudaFree(this->mData.p));
		}
	}
};

template<typename TType>
struct device_memory_3d
{
	device_memory_3d()
	{}

	device_memory_3d(size_t aLineWidth, size_t aHeight, size_t aDepth)
	{
		//CUGIL_CHECK_RESULT();
	}

	device_ptr<TType> mData;
};



}//namespace cugip
