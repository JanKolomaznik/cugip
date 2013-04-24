#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/exception.hpp>

namespace cugip {

template<typename TType>
struct device_ptr
{
	CUGIP_DECL_HYBRID
	device_ptr(): p(0) 
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	device_ptr(const device_ptr &aArg): p(aArg.p) 
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	device_ptr(TType *aArg): p(aArg) 
	{ /*empty*/ }

	CUGIP_DECL_DEVICE TType * 
	operator->()
	{ return p; }

	CUGIP_DECL_HYBRID device_ptr &
	operator=(const device_ptr &aArg)
	{ p = aArg.p; return *this; }

	CUGIP_DECL_HYBRID device_ptr &
	operator=(TType *aArg)
	{ p = aArg; return *this; }

	CUGIP_DECL_HYBRID
	operator bool() const
	{ return p != 0; }

	CUGIP_DECL_HYBRID device_ptr
	byte_offset(int aOffset) const
	{
		return device_ptr(reinterpret_cast<TType *>((reinterpret_cast<char *>(p) + aOffset)));
	}

	TType *p;
};

template<typename TType>
struct access_helper
{
	access_helper(device_ptr<TType> &aPtr): ptr(aPtr) {}

	device_ptr<TType> &ptr;
	mutable TType tmp;

	operator TType()const
	{
		CUGIP_CHECK_RESULT(cudaMemcpy(&tmp, ptr.p, sizeof(TType), cudaMemcpyDeviceToHost));
		return tmp;
	}
	const access_helper&
	operator=(const TType &aArg)const
	{ 
		tmp = aArg; 
		CUGIP_CHECK_RESULT(cudaMemcpy(ptr.p, &aArg, sizeof(TType), cudaMemcpyHostToDevice));
		return *this;
	}
};

template<typename TType>
access_helper<TType>
operator*(device_ptr<TType> &aPtr)
{
	CUGIP_ASSERT(aPtr);
	return access_helper<TType>(aPtr);
}

template<typename TType>
CUGIP_DECL_DEVICE TType &
operator*(device_ptr<TType> &aPtr)
{
	CUGIP_ASSERT(aPtr);
	return *aPtr.get();
}
//------------------------------------------------------------------------
template<typename TType>
struct const_device_ptr
{
	CUGIP_DECL_HYBRID
	const_device_ptr(): p(0) 
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	const_device_ptr(const device_ptr<TType> &aArg): p(aArg.p) 
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	const_device_ptr(const const_device_ptr<TType> &aArg): p(aArg.p) 
	{ /*empty*/ }

	CUGIP_DECL_DEVICE const TType * 
	operator->()
	{ return p; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(const device_ptr<TType> &aArg)
	{ p = aArg.p; return *this; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(const const_device_ptr &aArg)
	{ p = aArg.p; return *this; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(TType *aArg)
	{ p = aArg; return *this; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(const TType *aArg)
	{ p = aArg; return *this; }

	CUGIP_DECL_HYBRID
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
	typedef dim_traits<2>::coord_t coord_t;
	typedef TType value_type;

	device_memory_2d()
	{}

	device_memory_2d(device_ptr<TType> aPtr, size_t aWidth, size_t aHeight, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	device_memory_2d(device_ptr<TType> aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	device_memory_2d(const device_memory_2d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~device_memory_2d()
	{ }

	inline CUGIP_DECL_HYBRID value_type &
	operator[](coord_t aCoords)
	{
		value_type *row = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mData.p) + aCoords.template get<1>() * mPitch);
		return row[aCoords.template get<0>()];
	}

	inline CUGIP_DECL_HYBRID extents_t 
	dimensions() const
	{ return mExtents; }

	device_ptr<TType> mData;
	extents_t mExtents;
	size_t mPitch;
};

template<typename TType>
struct const_device_memory_2d
{
	typedef dim_traits<2>::extents_t extents_t;
	typedef dim_traits<2>::coord_t coord_t;
	typedef const TType value_type;

	const_device_memory_2d()
	{}

	const_device_memory_2d(const_device_ptr<TType> aPtr, size_t aWidth, size_t aHeight, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_device_memory_2d(const_device_ptr<TType> aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_device_memory_2d(const device_memory_2d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_device_memory_2d(const const_device_memory_2d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~const_device_memory_2d()
	{ }

	inline CUGIP_DECL_HYBRID value_type &
	operator[](coord_t aCoords)
	{
		value_type *row = reinterpret_cast<value_type *>(reinterpret_cast<const char *>(mData.p) + aCoords.template get<1>() * mPitch);
		return row[aCoords.template get<0>()];
	}


	CUGIP_DECL_HYBRID extents_t 
	dimensions() const
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
		CUGIP_CHECK_RESULT(cudaMallocPitch(&devPtr, &(this->mPitch), aWidth * sizeof(TType), aHeight));
		this->mExtents.set<0>(aWidth);
		this->mExtents.set<0>(aHeight);
		this->mData = reinterpret_cast<TType*>(devPtr);

		D_PRINT(boost::str(boost::format("GPU allocation: 2D memory - %1% items, %2% bytes pitch, %3% item size") 
					% this->mExtents
					% this->mPitch
					% sizeof(TType)));
	}

	device_memory_2d_owner(extents_t aExtents)
	{
		void *devPtr = NULL;
		CUGIP_CHECK_RESULT(cudaMallocPitch(&devPtr, &(this->mPitch), aExtents.get<0>() * sizeof(TType), aExtents.get<1>()));
		this->mExtents = aExtents;
		this->mData = reinterpret_cast<TType*>(devPtr);

		D_PRINT(boost::str(boost::format("GPU allocation: 2D memory - %1% items, %2% bytes pitch, %3% item size") 
					% this->mExtents
					% this->mPitch
					% sizeof(TType)));
	}

	~device_memory_2d_owner()
	{
		if (this->mData) {
			CUGIP_ASSERT_RESULT(cudaFree(this->mData.p));
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
		//CUGIP_CHECK_RESULT();
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
