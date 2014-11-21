#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/math.hpp>
#include <cugip/utils.hpp>
#include <cugip/exception.hpp>

namespace cugip {

template<typename TType>
struct device_base_ptr
{
	CUGIP_DECL_HYBRID
	device_base_ptr(TType *aPtr): p(aPtr)
	{ /*empty*/ }


	CUGIP_DECL_HYBRID
	operator bool() const
	{ return p != 0; }


	CUGIP_DECL_HYBRID TType *
	get() const
	{
		return p;
	}

	TType *p;
};


template<typename TType>
struct device_ptr : device_base_ptr<TType>
{
	CUGIP_DECL_HYBRID
	device_ptr(): device_base_ptr<TType>(0)
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	device_ptr(const device_ptr &aArg): device_base_ptr<TType>(aArg.p)
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	device_ptr(TType *aArg): device_base_ptr<TType>(aArg)
	{ /*empty*/ }

	CUGIP_DECL_DEVICE TType *
	operator->()
	{ return this->p; }

	CUGIP_DECL_HYBRID device_ptr &
	operator=(const device_ptr &aArg)
	{ this->p = aArg.p; return *this; }

	CUGIP_DECL_HYBRID device_ptr &
	operator=(TType *aArg)
	{ this->p = aArg; return *this; }

	/*CUGIP_DECL_HYBRID
	operator bool() const
	{ return p != 0; }*/

	CUGIP_DECL_HYBRID device_ptr
	byte_offset(int aOffset) const
	{
		return device_ptr(reinterpret_cast<TType *>((reinterpret_cast<char *>(this->p) + aOffset)));
	}

	CUGIP_DECL_DEVICE void
	assign_device(const TType &aValue)
	{
		*(this->p) = aValue;
	}

	CUGIP_DECL_HOST void
	assign_host(const TType &aValue)
	{
		TType tmp = aValue;
		CUGIP_CHECK_RESULT(cudaMemcpy(this->p, &tmp, sizeof(TType), cudaMemcpyHostToDevice));
	}

	CUGIP_DECL_HOST TType
	retrieve_host() const
	{
		TType tmp;
		CUGIP_CHECK_RESULT(cudaMemcpy(&tmp, this->p, sizeof(TType), cudaMemcpyDeviceToHost));
		return tmp;
	}


/*	CUGIP_DECL_HYBRID TType *
	get() const
	{
		return p;
	}

	TType *p;*/
};

template<typename TType>
struct access_helper
{
	CUGIP_DECL_HYBRID
	access_helper(const device_ptr<TType> &aPtr): ptr(aPtr) {}

	const device_ptr<TType> &ptr;

	CUGIP_DECL_HYBRID
	operator TType()const
	{
#ifdef __CUDACC__
		return *(ptr.p);
#else
		TType tmp;
		CUGIP_CHECK_RESULT(cudaMemcpy(&tmp, ptr.p, sizeof(TType), cudaMemcpyDeviceToHost));
		return tmp;
#endif
	}

	CUGIP_DECL_HYBRID
	const access_helper&
	operator=(const TType &aArg)const
	{
#ifdef __CUDACC__
		*(ptr.p) = aArg;
#else
		TType tmp = aArg;
		CUGIP_CHECK_RESULT(cudaMemcpy(ptr.p, &aArg, sizeof(TType), cudaMemcpyHostToDevice));
#endif
		return *this;
	}
};

template<typename TType>
CUGIP_DECL_HYBRID access_helper<TType>
operator*(const device_ptr<TType> &aPtr)
{
	CUGIP_ASSERT(aPtr);
	return access_helper<TType>(aPtr);
}


/*#ifdef __CUDACC__
template<typename TType>
CUGIP_DECL_DEVICE TType &
operator*(const device_ptr<TType> &aPtr)
{
	CUGIP_ASSERT(aPtr);
	return *aPtr.get();
}
#else
template<typename TType>
access_helper<TType>
operator*(const device_ptr<TType> &aPtr)
{
	CUGIP_ASSERT(aPtr);
	return access_helper<TType>(aPtr);
}
#endif // __CUDACC__
*/
//------------------------------------------------------------------------
template<typename TType>
struct const_device_ptr : device_base_ptr<const TType>
{
	CUGIP_DECL_HYBRID
	const_device_ptr(): device_base_ptr<const TType>(0)
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	const_device_ptr(const device_ptr<TType> &aArg): device_base_ptr<const TType>(aArg.p)
	{ /*empty*/ }

	CUGIP_DECL_HYBRID
	const_device_ptr(const const_device_ptr<TType> &aArg): device_base_ptr<const TType>(aArg.p)
	{ /*empty*/ }

	CUGIP_DECL_DEVICE const TType *
	operator->()
	{ return this->p; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(const device_ptr<TType> &aArg)
	{ this->p = aArg.p; return *this; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(const const_device_ptr &aArg)
	{ this->p = aArg.p; return *this; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(TType *aArg)
	{ this->p = aArg; return *this; }

	CUGIP_DECL_HYBRID const_device_ptr &
	operator=(const TType *aArg)
	{ this->p = aArg; return *this; }

	/*CUGIP_DECL_HYBRID
	operator bool() const
	{ return p != 0; }

	CUGIP_DECL_HYBRID const TType *
	get() const
	{
		return p;
	}

	const TType *p;*/
};

template<typename TType>
struct device_memory_1d
{
	typedef dim_traits<1>::extents_t extents_t;
	typedef dim_traits<1>::coord_t coord_t;
	typedef TType value_type;

	device_memory_1d()
	{}

	device_memory_1d(device_ptr<TType> aPtr, size_t aSize)
		:mData(aPtr), mExtents(aSize)
	{}

	device_memory_1d(device_ptr<TType> aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents)
	{}

	device_memory_1d(const device_memory_1d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents)
	{}

	~device_memory_1d()
	{ }

	inline CUGIP_DECL_HYBRID value_type &
	operator[](coord_t aCoords)
	{
		return mData.p[aCoords[0]];
	}

	inline CUGIP_DECL_HYBRID value_type &
	operator[](size_t aIdx)
	{
		return mData.p[aIdx];
	}

	inline CUGIP_DECL_HYBRID const value_type &
	operator[](coord_t aCoords) const
	{
		return mData.p[aCoords[0]];
	}

	inline CUGIP_DECL_HYBRID const value_type &
	operator[](size_t aIdx) const
	{
		return mData.p[aIdx];
	}

	inline CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mExtents; }

	device_ptr<TType> mData;
	extents_t mExtents;
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
	operator[](coord_t aCoords) const
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
	operator[](coord_t aCoords) const
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
struct device_memory_3d
{
	typedef dim_traits<3>::extents_t extents_t;
	typedef dim_traits<3>::coord_t coord_t;
	typedef TType value_type;

	device_memory_3d()
	{}

	device_memory_3d(device_ptr<TType> aPtr, size_t aWidth, size_t aHeight,  size_t aDepth, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight, aDepth), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	device_memory_3d(device_ptr<TType> aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	device_memory_3d(const device_memory_3d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}


	~device_memory_3d()
	{ }

	inline CUGIP_DECL_HYBRID value_type &
	operator[](coord_t aCoords) const
	{
		int offset = aCoords[0] * sizeof(value_type)
		             + aCoords[1] * mPitch
		             + aCoords[2] * mPitch * mExtents[1];
		return *reinterpret_cast<value_type *>(reinterpret_cast<char *>(mData.p) + offset);
	}

	inline CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mExtents; }

	device_ptr<TType> mData;
	extents_t mExtents;
	size_t mPitch;
};


template<typename TType>
struct const_device_memory_3d
{
	typedef dim_traits<3>::extents_t extents_t;
	typedef dim_traits<3>::coord_t coord_t;
	typedef const TType value_type;

	const_device_memory_3d()
	{}

	const_device_memory_3d(device_ptr<TType> aPtr, size_t aWidth, size_t aHeight,  size_t aDepth, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight, aDepth), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_device_memory_3d(device_ptr<TType> aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_device_memory_3d(const device_memory_3d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_device_memory_3d(const const_device_memory_3d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~const_device_memory_3d()
	{ }

	inline CUGIP_DECL_HYBRID value_type &
	operator[](coord_t aCoords) const
	{
		int offset = aCoords[0] * sizeof(value_type)
		             + aCoords[1] * mPitch
		             + aCoords[2] * mPitch * mExtents[1];
		return *reinterpret_cast<value_type *>(reinterpret_cast<const char *>(mData.p) + offset);
	}

	inline CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mExtents; }

	const_device_ptr<TType> mData;
	extents_t mExtents;
	size_t mPitch;
};



template<typename TType>
struct device_memory_1d_owner: public device_memory_1d<TType>
{
	typedef dim_traits<1>::extents_t extents_t;

	device_memory_1d_owner()
	{}

	device_memory_1d_owner(size_t aSize)
	{
		void *devPtr = NULL;
		CUGIP_CHECK_RESULT(cudaMalloc(&devPtr, aSize * sizeof(TType)));
		this->mExtents.template set<0>(aSize);
		this->mData = reinterpret_cast<TType*>(devPtr);

		D_PRINT(boost::str(boost::format("GPU allocation: 1D memory - %1% items, %2% item size")
					% this->mExtents
					% sizeof(TType)));
	}

	device_memory_1d_owner(extents_t aExtents)
	{
		void *devPtr = NULL;
		CUGIP_CHECK_RESULT(cudaMalloc(&devPtr, aExtents.get<0>() * sizeof(TType)));
		this->mExtents = aExtents;
		this->mData = reinterpret_cast<TType*>(devPtr);

		D_PRINT(boost::str(boost::format("GPU allocation: 1D memory - %1% items, %2% item size")
					% this->mExtents
					% sizeof(TType)));
	}

	~device_memory_1d_owner()
	{
		if (this->mData) {
			CUGIP_ASSERT_RESULT(cudaFree(this->mData.p));
		}
	}
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
		this->mExtents.set<1>(aHeight);
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
struct device_memory_3d_owner: public device_memory_3d<TType>
{
	typedef dim_traits<3>::extents_t extents_t;

	device_memory_3d_owner()
	{}

	device_memory_3d_owner(size_t aWidth, size_t aHeight, size_t aDepth)
	{
		cudaPitchedPtr pitchedDevPtr;

		CUGIP_CHECK_RESULT(cudaMalloc3D(&pitchedDevPtr, make_cudaExtent(aWidth * sizeof(TType), aHeight, aDepth)));
		this->mExtents.set<0>(aWidth);
		this->mExtents.set<1>(aHeight);
		this->mExtents.set<2>(aDepth);
		this->mData = reinterpret_cast<TType*>(pitchedDevPtr.ptr);
		this->mPitch = pitchedDevPtr.pitch;
		D_PRINT(boost::str(boost::format("GPU allocation: 3D memory - %1% items, %2% bytes pitch, %3% item size")
					% this->mExtents
					% this->mPitch
					% sizeof(TType)));
	}

	device_memory_3d_owner(extents_t aExtents)
	{
		cudaPitchedPtr pitchedDevPtr;
		CUGIP_CHECK_RESULT(cudaMalloc3D(&pitchedDevPtr, make_cudaExtent(aExtents[0] * sizeof(TType), aExtents[1], aExtents[2])));
		this->mExtents = aExtents;
		this->mData = reinterpret_cast<TType*>(pitchedDevPtr.ptr);
		this->mPitch = pitchedDevPtr.pitch;

		D_PRINT(boost::str(boost::format("GPU allocation: 3D memory - %1% items, %2% bytes pitch, %3% item size")
					% this->mExtents
					% this->mPitch
					% sizeof(TType)));

		CUGIP_CHECK_RESULT(cudaMemset(pitchedDevPtr.ptr, 0, this->mPitch * aExtents[1] * aExtents[2]));
	}

	~device_memory_3d_owner()
	{
		if (this->mData) {
			CUGIP_ASSERT_RESULT(cudaFree(this->mData.p));
		}
	}
};
//*****************************************************************************************

template<typename TType>
struct host_memory_2d
{
	typedef dim_traits<2>::extents_t extents_t;
	typedef dim_traits<2>::coord_t coord_t;
	typedef TType value_type;

	host_memory_2d()
	{}

	host_memory_2d(TType *aPtr, size_t aWidth, size_t aHeight, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	host_memory_2d(TType *aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	host_memory_2d(const host_memory_2d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~host_memory_2d()
	{ }

	value_type &
	operator[](coord_t aCoords) const
	{
		value_type *row = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mData) + aCoords.template get<1>() * mPitch);
		return row[aCoords.template get<0>()];
	}

	extents_t
	dimensions() const
	{ return mExtents; }

	TType *mData;
	extents_t mExtents;
	size_t mPitch;
};

template<typename TType>
struct const_host_memory_2d
{
	typedef dim_traits<2>::extents_t extents_t;
	typedef dim_traits<2>::coord_t coord_t;
	typedef const TType value_type;

	const_host_memory_2d()
	{}

	const_host_memory_2d(const TType *aPtr, size_t aWidth, size_t aHeight, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_host_memory_2d(const TType *aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_host_memory_2d(const host_memory_2d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_host_memory_2d(const const_host_memory_2d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~const_host_memory_2d()
	{ }

	value_type &
	operator[](coord_t aCoords) const
	{
		value_type *row = reinterpret_cast<value_type *>(reinterpret_cast<const char *>(mData) + aCoords.template get<1>() * mPitch);
		return row[aCoords.template get<0>()];
	}


	extents_t
	dimensions() const
	{ return mExtents; }

	const TType *mData;
	extents_t mExtents;
	size_t mPitch;
};


template<typename TType>
struct host_memory_3d
{
	typedef dim_traits<3>::extents_t extents_t;
	typedef dim_traits<3>::coord_t coord_t;
	typedef TType value_type;

	host_memory_3d()
	{}

	host_memory_3d(TType *aPtr, size_t aWidth, size_t aHeight,  size_t aDepth, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight, aDepth), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	host_memory_3d(TType *aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	host_memory_3d(const host_memory_3d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}


	~host_memory_3d()
	{ }

	value_type &
	operator[](coord_t aCoords) const
	{
		value_type *row = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mData) + aCoords.template get<1>() * aCoords.template get<2>() * mPitch);
		return row[aCoords.template get<0>()];
	}

	extents_t
	dimensions() const
	{ return mExtents; }

	TType *mData;
	extents_t mExtents;
	size_t mPitch;
};


template<typename TType>
struct const_host_memory_3d
{
	typedef dim_traits<3>::extents_t extents_t;
	typedef dim_traits<3>::coord_t coord_t;
	typedef const TType value_type;

	const_host_memory_3d()
	{}

	const_host_memory_3d(const TType *aPtr, size_t aWidth, size_t aHeight,  size_t aDepth, size_t aPitch)
		:mData(aPtr), mExtents(aWidth, aHeight, aDepth), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_host_memory_3d(const TType *aPtr, extents_t aExtents, size_t aPitch)
		:mData(aPtr), mExtents(aExtents), mPitch(aPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_host_memory_3d(const host_memory_3d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	const_host_memory_3d(const const_host_memory_3d<TType> &aMemory)
		:mData(aMemory.mData), mExtents(aMemory.mExtents), mPitch(aMemory.mPitch)
	{
		CUGIP_ASSERT(mPitch >= (mExtents.get<0>()*sizeof(TType)));
	}

	~const_host_memory_3d()
	{ }

	value_type &
	operator[](coord_t aCoords) const
	{
		value_type *row = reinterpret_cast<value_type *>(reinterpret_cast<char *>(mData) + aCoords.template get<1>() * aCoords.template get<2>() * mPitch);
		return row[aCoords.template get<0>()];
	}

	extents_t
	dimensions() const
	{ return mExtents; }

	const TType *mData;
	extents_t mExtents;
	size_t mPitch;
};

//*****************************************************************************************

template<typename TElement, size_t tDim>
struct memory_management;

template<typename TElement>
struct memory_management<TElement, 2>
{
	typedef device_memory_2d<TElement> device_memory;
	typedef const_device_memory_2d<TElement> const_device_memory;
	typedef device_memory_2d_owner<TElement> device_memory_owner;

	typedef host_memory_2d<TElement> host_memory;
	typedef const_host_memory_2d<TElement> const_host_memory;
};

template<typename TElement>
struct memory_management<TElement, 3>
{
	typedef device_memory_3d<TElement> device_memory;
	typedef const_device_memory_3d<TElement> const_device_memory;
	typedef device_memory_3d_owner<TElement> device_memory_owner;

	typedef host_memory_3d<TElement> host_memory;
	typedef const_host_memory_3d<TElement> const_host_memory;
};

template <typename TType>
device_memory_1d<TType>
view(const device_memory_1d_owner<TType> &aBuffer)
{
	return static_cast<const device_memory_1d<TType> &>(aBuffer);
}


}//namespace cugip
