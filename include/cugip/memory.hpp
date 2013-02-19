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
struct const_device_ptr
{
	CUGIL_DECL_HYBRID
	const_device_ptr(): p(0) 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID
	const_device_ptr(const device_ptr<TType> &aArg): p(aArg.p) 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID
	const_device_ptr(const const_device_ptr<TType> &aArg): p(aArg.p) 
	{ /*empty*/ }

	CUGIL_DECL_HYBRID const TType * 
	operator->()
	{ return p; }

	CUGIL_DECL_HYBRID const_device_ptr &
	operator=(const device_ptr<TType> &aArg)
	{ p = aArg.p; return *this; }

	CUGIL_DECL_HYBRID const_device_ptr &
	operator=(const const_device_ptr &aArg)
	{ p = aArg.p; return *this; }

	CUGIL_DECL_HYBRID const_device_ptr &
	operator=(TType *aArg)
	{ p = aArg; return *this; }

	CUGIL_DECL_HYBRID const_device_ptr &
	operator=(const TType *aArg)
	{ p = aArg; return *this; }

	CUGIL_DECL_HYBRID
	operator bool() const
	{ return p != 0; }

	const TType *p;
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
		:mData(aPtr), mExtents(aWidth, aHeight), mPitch(aPitch)
	{
		CUGIL_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	device_memory_2d(device_ptr<TType> aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIL_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~device_memory_2d()
	{ }

	extents_t size() const
	{ return mExtents; }

	device_ptr<TType> mData;
	extents_t mExtents;
	size_t mPitch;
};

template<typename TType>
struct const_device_memory_2d
{
	typedef dim_traits<2>::extents_t extents_t;

	const_device_memory_2d()
	{}

	const_device_memory_2d(const_device_ptr<TType> aPtr, size_t aWidth, size_t aHeight, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight), mPitch(aPitch)
	{
		CUGIL_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_device_memory_2d(const_device_ptr<TType> aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIL_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~const_device_memory_2d()
	{ }

	extents_t size() const
	{ return mExtents; }

	const_device_ptr<TType> mData;
	extents_t mExtents;
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
		this->mExtents.set<0>(aWidth);
		this->mExtents.set<0>(aHeight);
		this->mData = reinterpret_cast<TType*>(devPtr);
	}

	device_memory_2d_owner(extents_t aExtents)
	{
		void *devPtr = NULL;
		CUGIL_CHECK_RESULT(cudaMallocPitch(&devPtr, &(this->mPitch), aExtents.get<0>() * sizeof(TType), aExtents.get<1>()));
		this->mExtents = aExtents;
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
	typedef const_device_memory_2d<TElement> const_device_memory;
	typedef device_memory_2d_owner<TElement> device_memory_owner;
};

}//namespace cugip
