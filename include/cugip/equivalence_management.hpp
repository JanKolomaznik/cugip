#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>

//#include <thrust/reduce.h>
//#include <thrust/execution_policy.h>

namespace cugip {

template <typename TClassId>
CUGIP_GLOBAL void
kernelCompaction(TClassId *aBuffer, int64_t aSize)
{
	int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int64_t stride = gridDim.x * blockDim.x;
	//int pass = 0;
	while (idx < aSize) {
		int64_t newValue = idx;
		while (aBuffer[newValue] != newValue) {
			newValue = aBuffer[newValue];
			CUGIP_ASSERT(newValue >= 0 && newValue < aSize);
			assert(newValue >= 0 && newValue < aSize);
		}
		aBuffer[idx] = newValue;
		idx += stride;
		/*++pass;
		if (threadIdx.x == 0) {
			printf("Pass %d", pass);
		}*/
	}
}

template <typename TClassId>
CUGIP_GLOBAL void
kernelTest(TClassId *aBuffer, int64_t aSize)
{
	if (threadIdx.x == 0) {
		int sum = 0;
		for (int i = 0; i < 1; ++i) {
			sum += aBuffer[i];
		}
	}
}

template <typename TClassId>
CUGIP_GLOBAL void
kernelInitialization(TClassId *aBuffer, int64_t aSize)
{
	int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < aSize) {
		aBuffer[idx] = idx;
	}
}


template<typename TClassId>
class EquivalenceManager
{
public:
	explicit EquivalenceManager(TClassId *aBuffer = nullptr, int64_t aClassCount = 0)
		: mBuffer(aBuffer)
		, mSize(aClassCount)
	{
		CUGIP_DFORMAT("EquivalenceManager buffer ptr: %p size: %d", uintptr_t(mBuffer), mSize);
	}

	CUGIP_DECL_HYBRID
	EquivalenceManager(const EquivalenceManager &aOther)
		: mBuffer(aOther.mBuffer)
		, mSize(aOther.mSize)
	{}

	CUGIP_DECL_HYBRID
	EquivalenceManager &
	operator=(const EquivalenceManager &aOther)
	{
		mBuffer = aOther.mBuffer;
		mSize = aOther.mSize;
		return *this;
	}

	CUGIP_DECL_DEVICE
	TClassId
	get(TClassId aClass) const
	{
		return mBuffer[aClass];
	}

	CUGIP_DECL_DEVICE
	TClassId
	merge(TClassId aFirst, TClassId aSecond)
	{
		CUGIP_ASSERT(mBuffer != nullptr);
		CUGIP_ASSERT(mSize > 0);
		CUGIP_ASSERT(aFirst < mSize);
		CUGIP_ASSERT(aSecond < mSize);
		TClassId minId = min(aFirst, aSecond);
		TClassId maxId = max(aFirst, aSecond);
		//TClassId minRoot = mBuffer[minId];
		TClassId maxRoot = mBuffer[maxId];
		if (maxRoot > minId) {
		//if (maxRoot > minRoot) {
			//mBuffer[maxId] = minRoot;
			mBuffer[maxId] = minId;
		}
		//return minRoot;
		return minId;
	}

	CUGIP_DECL_HOST
	void compaction()
	{
		CUGIP_ASSERT(mBuffer != nullptr);
		CUGIP_ASSERT(mSize > 0);
		dim3 blockSize(256, 1, 1);
		dim3 gridSize(1, 1, 1);
		//dim3 gridSize((mSize + 255) / 256, 1, 1);
		CUGIP_CHECK_ERROR_STATE("before compaction kernel");
		//auto t = thrust::reduce(thrust::device, mBuffer, mBuffer + mSize);

		kernelTest<TClassId><<<1, 32>>>(mBuffer, mSize);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());

		CUGIP_DFORMAT("Before compaction EquivalenceManager buffer ptr: %p size: %d", uintptr_t(mBuffer), mSize);
		kernelCompaction<TClassId><<<gridSize, blockSize>>>(mBuffer, mSize);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	CUGIP_DECL_HOST
	void initialize()
	{
		CUGIP_ASSERT(mBuffer != nullptr);
		CUGIP_ASSERT(mSize > 0);
		dim3 blockSize(256, 1, 1);
		dim3 gridSize((mSize + 255) / 256, 1, 1);

		kernelInitialization<TClassId><<<gridSize, blockSize>>>(mBuffer, mSize);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}
protected:
	TClassId *mBuffer;
	int64_t mSize;
};

} // namespace cugip
