#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>
#include <cugip/math.hpp>

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
	{ p = aArg.p; return *this; }

	CUGIL_DECL_HYBRID device_ptr &
	operator=(TType *aArg)
	{ p = aArg; return *this; }

	CUGIL_DECL_HYBRID
	operator bool() const
	{ return p != 0; }

	TType *p;
};

template<typename TType>
struct device_memory_1d
{
	
};

//****************************************************
template<typename TType>
struct device_memory_2d
{
	typedef dim_traits<2>::extents_t extents_t;

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
	typedef dim_traits<2>::extents_t extents_t;

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

	device_memory_2d_owner(extents_t aExtents)
	{
		void *devPtr = NULL;
		CUGIL_CHECK_RESULT(cudaMallocPitch(&devPtr, &(this->mPitch), aExtents.get<0>() * sizeof(TType), aExtents.get<1>()));
		this->mWidth = aExtents.get<0>();
		this->mHeight = aExtents.get<1>();
		this->mData = reinterpret_cast<TType*>(devPtr);
	}

	~device_memory_2d_owner()
	{
		if (this->mData) {
			CUGIL_ASSERT_RESULT(cudaFree(this->mData.p));
		}
	}
};

//****************************************************

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

template<typename TElement, size_t tDim>
struct memory_management;

template<typename TElement>
struct memory_management<TElement, 2>
{
	typedef device_memory_2d<TElement> device_memory;
	typedef device_memory_2d_owner<TElement> device_memory_owner;
};

}//namespace cugip
