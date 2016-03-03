#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/device_flag.hpp>
#include <limits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>


#include <boost/filesystem.hpp>
#include <fstream>


namespace cugip {

template<typename TType>
class ParallelQueueView
{
public:
	CUGIP_DECL_DEVICE int
	allocate(int aItemCount)
	{
		return atomicAdd(mSize.get(), aItemCount);
	}

	CUGIP_DECL_DEVICE int
	append(const TType &aItem)
	{
		int index = atomicAdd(mSize.get(), 1);
		mData[index] = aItem;
		return index;
	}

	CUGIP_DECL_DEVICE int
	push_back(const TType &aItem)
	{
		return append(aItem);
	}

	CUGIP_DECL_HYBRID int
	size()
	{
		#if __CUDA_ARCH__
			return device_size();
		#else
			return host_size();
		#endif
	}

	CUGIP_DECL_HOST int
	host_size()
	{
		cudaThreadSynchronize();
		return mSize.retrieve_host();
	}

	CUGIP_DECL_DEVICE int
	device_size()
	{
		return mSize.retrieve_device();
	}

	CUGIP_DECL_HOST void
	clear()
	{
		mSize.assign_host(0);
	}

	CUGIP_DECL_DEVICE void
	clear_device()
	{
		mSize.assign_device(0);
	}

	CUGIP_DECL_DEVICE TType &
	operator[](int aIndex)
	{
		return get_device(aIndex);
	}

	CUGIP_DECL_DEVICE TType &
	back()
	{
		CUGIP_ASSERT(*mSize > 0);
		return get_device((*mSize) - 1);
	}


	CUGIP_DECL_DEVICE TType &
	get_device(int aIndex)
	{
		/*assert(aIndex >= 0);
		assert(aIndex < *mSize);*/
		//if( aIndex >= mSize.retrieve_device()) printf("ERRRROR %d - %d\n", aIndex, mSize.retrieve_device());
		return mData[aIndex];
	}

	CUGIP_DECL_DEVICE TType &
	get_device(int aIndex) const
	{
		return mData[aIndex];
	}

	TType *mData;
	device_ptr<int> mSize;
};


template<typename TType>
class ParallelQueue
{
public:
	ParallelQueue()
		: mSizePointer(1)
	{
		mView.mSize = mSizePointer.mData;
	}

	ParallelQueueView<TType> &
	view()
	{
		return mView;
	}

	int
	size()
	{
		cudaThreadSynchronize();
		return mView.size();
	}

	void
	clear()
	{
		mView.clear();
	}

	void
	reserve(int aSize)
	{
		if (aSize > mBuffer.size()) {
			mBuffer.resize(aSize);
			mView.mData = thrust::raw_pointer_cast(&(mBuffer[0]));
		}
	}

	void
	fill_host(thrust::host_vector<TType> &v)
	{
		int s = size();
		v.resize(s);
		thrust::copy(mBuffer.begin(), mBuffer.begin() + s, v.begin());
	}
	void
	fill_host(std::vector<TType> &v)
	{
		int s = size();
		v.resize(s);
		thrust::copy(mBuffer.begin(), mBuffer.begin() + s, v.data());
	}

//protected:
	ParallelQueue(const ParallelQueue &);

	ParallelQueueView<TType> mView;
	thrust::device_vector<TType> mBuffer;
	device_memory_1d_owner<int> mSizePointer;
};


} // namespace cugip
