#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>

namespace cugip {

template <typename TClassId>
CUGIP_GLOBAL void
kernelCompaction(TClassId *aBuffer, int aSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < aSize) {
		int newValue = idx;
		while (aBuffer[newValue] != newValue) {
			newValue = aBuffer[newValue];
		}
		aBuffer[idx] = newValue;
	}
}

template <typename TClassId>
CUGIP_GLOBAL void
kernelInitialization(TClassId *aBuffer, int aSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < aSize) {
		aBuffer[idx] = idx;
	}
}


template<typename TClassId>
class EquivalenceManager
{
public:
	EquivalenceManager(TClassId *aBuffer = nullptr, int aClassCount = 0)
		: mBuffer(aBuffer)
		, mSize(aClassCount)
	{}

	CUGIP_DECL_DEVICE
	TClassId
	get(TClassId aClass) const
	{
		return mBuffer[aClass];
	}

	CUGIP_DECL_DEVICE
	void
	merge(TClassId aFirst, TClassId aSecond)
	{
		TClassId minId = min(aFirst, aSecond);
		TClassId maxId = max(aFirst, aSecond);
		if (mBuffer[maxId] > minId) {
			mBuffer[maxId] = minId;
		}
	}

	CUGIP_DECL_HOST
	void compaction()
	{
		dim3 blockSize(256, 1, 1);
		dim3 gridSize((mSize + 255) / 256, 1, 1);

		kernelCompaction<TClassId><<<gridSize, blockSize>>>(mBuffer, mSize);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	CUGIP_DECL_HOST
	void initialize()
	{
		dim3 blockSize(256, 1, 1);
		dim3 gridSize((mSize + 255) / 256, 1, 1);

		kernelInitialization<TClassId><<<gridSize, blockSize>>>(mBuffer, mSize);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}
protected:
	TClassId *mBuffer;
	int mSize;
};

} // namespace cugip
