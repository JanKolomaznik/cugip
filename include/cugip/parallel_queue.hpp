#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/array_view.hpp>
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
		CUGIP_ASSERT(device_size() < mSizeLimit);
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
	size() const
	{
		#if __CUDA_ARCH__
			return device_size();
		#else
			return host_size();
		#endif
	}

	CUGIP_DECL_HOST int
	host_size() const
	{
		cudaDeviceSynchronize();
		return mSize.retrieve_host();
	}

	CUGIP_DECL_DEVICE int
	device_size() const
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
		/*CUGIP_ASSERT(*mSize > 0);
		return get_device((*mSize) - 1);*/
		CUGIP_ASSERT(mSize.retrieve_device() > 0);
		return get_device((mSize.retrieve_device()) - 1);
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

	CUGIP_DECL_HYBRID
	TType *pointer() const {
		return mData;
	}

	DeviceArrayView<TType> array_view() {
		return DeviceArrayView<TType>(this->pointer(), this->size());
	}

	DeviceArrayConstView<TType> const_array_view() const {
		return DeviceArrayConstView<TType>(this->pointer(), this->size());
	}

	TType *mData;
	device_ptr<int> mSize;
	int mSizeLimit = 0;
};


template<typename TType>
class ParallelQueue
{
public:
	ParallelQueue()
		: mSizePointer(1)
	{
		mView.mSize = mSizePointer.mData;
		clear();
	}

	ParallelQueueView<TType> &
	view()
	{
		return mView;
	}

	int
	size() const
	{
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
			mView.mSizeLimit = mBuffer.size();
		}
	}

	void
	resize(int aSize)
	{
		// TODO - check for correctness
		device_ptr<int> size = mSizePointer.mData;
		size.assign_host(aSize);
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

	DeviceArrayView<TType> array_view() {
		return mView.array_view();
	}

	DeviceArrayConstView<TType> const_array_view() const {
		return mView.const_array_view();

	}

//protected:
	ParallelQueue(const ParallelQueue &);
	//TODO move constructor

	ParallelQueueView<TType> mView;
	thrust::device_vector<TType> mBuffer;
	device_memory_1d_owner<int> mSizePointer;
};

template<typename TType>
auto view(ParallelQueue<TType> &aQueue) {
	return aQueue.view();
}

template<typename TType>
auto const_view(ParallelQueue<TType> &aQueue) {
	// TODO do we need special const view?
	return aQueue.view();
}

} // namespace cugip
