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

#define INVALID_LABEL (1 << 30)

namespace cugip {

template<typename TFlow>
struct EdgeResidualsRecord
{
	__host__ __device__
	EdgeResidualsRecord( TFlow aWeight = 0.0f )
	{
		residuals[0] = residuals[1] = aWeight;
	}

	__host__ __device__ float &
	getResidual( bool aFirst )
	{
		return aFirst ? residuals[0] : residuals[1];
	}
	TFlow residuals[2];
};


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

	CUGIP_DECL_HOST int
	size()
	{
		cudaThreadSynchronize();
		return mSize.retrieve_host();
	}

	CUGIP_DECL_HOST void
	clear()
	{
		mSize.assign_host(0);
	}


	CUGIP_DECL_DEVICE TType &
	get_device(int aIndex)
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
		mBuffer.resize(aSize);
		mView.mData = thrust::raw_pointer_cast(&(mBuffer[0]));
	}
protected:
	ParallelQueueView<TType> mView;
	thrust::device_vector<TType> mBuffer;
	device_memory_1d_owner<int> mSizePointer;
};

template<typename TFlow>
struct GraphCutData
{
	CUGIP_DECL_DEVICE int
	neighborCount(int aVertexId)
	{
		return firstNeighborIndex(aVertexId + 1) - firstNeighborIndex(aVertexId);
	}

	CUGIP_DECL_DEVICE TFlow &
	excess(int aVertexId)
	{
		return vertexExcess[aVertexId];
	}

	CUGIP_DECL_DEVICE int &
	label(int aVertexId)
	{
		return labels[aVertexId];
	}

	CUGIP_DECL_HYBRID int
	vertexCount()
	{
		return mVertexCount;
	}

	CUGIP_DECL_DEVICE int
	firstNeighborIndex(int aVertexId)
	{
		return neighbors[aVertexId];
	}

	CUGIP_DECL_DEVICE int
	secondVertex(int aIndex)
	{
		return secondVertices[aIndex];
	}

	CUGIP_DECL_DEVICE int
	connectionIndex(int aIndex)
	{
		return connectionIndices[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow
	sourceTLinkCapacity(int aIndex)
	{
		return mSourceTLinks[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow
	sinkTLinkCapacity(int aIndex)
	{
		return mSinkTLinks[aIndex];
	}

	CUGIP_DECL_DEVICE EdgeResidualsRecord<TFlow> &
	residuals(int aIndex)
	{
		return mResiduals[aIndex];
	}


	int mVertexCount;

	TFlow *vertexExcess; // n
	int *labels; // n
	int *neighbors; // n

	TFlow *mSourceTLinks; // n
	TFlow *mSinkTLinks; // n

	int *secondVertices; // 2 * m
	int *connectionIndices; // 2 * m
	EdgeResidualsRecord<TFlow> *mResiduals; // m

};


template<typename TFlow>
CUGIP_GLOBAL void
initBFSKernel(ParallelQueueView<int> aVertices, GraphCutData<TFlow> aGraph)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		int label = INVALID_LABEL;
			//printf("checking %d - %d\n", index, aGraph.vertexCount());
		if (aGraph.sinkTLinkCapacity(index) > 0.0) {
			label = 1;
			aVertices.append(index);
			//printf("adding %d\n", index);
		}
		aGraph.label(index) = label;
	}
}

template<typename TFlow>
CUGIP_GLOBAL void
bfsPropagationKernel(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		GraphCutData<TFlow> aGraph,
		int aCurrentLevel/*,
		device_flag_view aPropagationSuccessfulFlag*/)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aCount) {
		int vertex = aVertices.get_device(aStart + index);
		int neighborCount = aGraph.neighborCount(vertex);
		int firstNeighborIndex = aGraph.firstNeighborIndex(vertex);
		for (int i = 0; i < neighborCount; ++i) {
			int secondVertex = aGraph.secondVertex(firstNeighborIndex + i);
				//printf("%d - %d\n", vertex, secondVertex);
			int label = aGraph.label(secondVertex);
			TFlow residual = aGraph.residuals(aGraph.connectionIndex(firstNeighborIndex + i)).getResidual(vertex > secondVertex);
			if (vertex == 210652) {
				printf("aaaa - %d - %d %d\n", vertex, secondVertex, firstNeighborIndex + i);
			}
			if (label == INVALID_LABEL && residual > 0.0f) {
				aGraph.label(secondVertex) = aCurrentLevel; //TODO atomic
				aVertices.append(secondVertex);
				//printf("%d\n", secondVertex);
			}
			//printf("%d, %d, %d\n", vertex, secondVertex, label);
		}
		//printf("%d, %d\n", vertex, neighborCount);
		//printf("%d %d %d\n", index, blockId, aCount);
	}
}

template<typename TFlow>
CUGIP_DECL_DEVICE TFlow
tryPull(GraphCutData<TFlow> &aGraph, int aFrom, int aTo, TFlow aCapacity)
{
	TFlow excess = aGraph.excess(aFrom);
	while (excess > 0.0f && aCapacity > 0.0f) {
		float pushedFlow = min(excess, aCapacity);
		//printf("%d %d %f\n", aFrom, aTo, aCapacity);
		TFlow oldExcess = atomicFloatCAS(&(aGraph.excess(aFrom)), excess, excess - pushedFlow);
		if(excess == oldExcess) {
			return pushedFlow;
		} else {
			excess = oldExcess;
		}
	}

	return 0;
}


template<typename TFlow>
CUGIP_DECL_DEVICE bool
tryPullPush(GraphCutData<TFlow> &aGraph, int aFrom, int aTo, int aConnectionIndex)
{
	EdgeResidualsRecord<TFlow> &edge = aGraph.residuals(aConnectionIndex);
	TFlow residual = edge.getResidual(aFrom < aTo);
	TFlow flow = tryPull(aGraph, aFrom, aTo, residual);
	if (flow > 0) {
		atomicAdd(&(aGraph.excess(aTo)), flow);
		edge.getResidual(aFrom < aTo) -= flow;
		edge.getResidual(aFrom > aTo) += flow;
		return true;
	}
	return false;
}
/* // Each iteration visits all vertices
template<typename TFlow>
CUGIP_GLOBAL void
pushKernel(
		GraphCutData<TFlow> aGraph,
		device_flag_view aPushSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		int neighborCount = aGraph.neighborCount(index);
		int firstNeighborIndex = aGraph.firstNeighborIndex(index);
		int label = aGraph.label(index);
		for (int i = 0; i < neighborCount; ++i) {
			int secondVertex = aGraph.secondVertex(firstNeighborIndex + i);
			int secondLabel = aGraph.label(secondVertex);
			if (label > secondLabel) {
				int connectionIndex = aGraph.connectionIndex(firstNeighborIndex + i);
				if (tryPullPush(aGraph, index, secondVertex, connectionIndex)) {
					aPushSuccessfulFlag.set_device();
				}
			}
		}
	}
}*/

template<typename TFlow>
CUGIP_GLOBAL void
pushKernel(
		GraphCutData<TFlow> aGraph,
		ParallelQueueView<int> aVertices,
		int aLevelStart,
		int aLevelEnd,
		device_flag_view aPushSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = aLevelStart + blockId * blockDim.x + threadIdx.x;

	if (index < aLevelEnd) {
		int vertex = aVertices.get_device(index);
		if (aGraph.excess(vertex) > 0.0f) {
			int neighborCount = aGraph.neighborCount(vertex);
			int firstNeighborIndex = aGraph.firstNeighborIndex(vertex);
			int label = aGraph.label(vertex);
			for (int i = 0; i < neighborCount; ++i) {
				int secondVertex = aGraph.secondVertex(firstNeighborIndex + i);
				int secondLabel = aGraph.label(secondVertex);
				if (label > secondLabel) {
					int connectionIndex = aGraph.connectionIndex(firstNeighborIndex + i);
					if (tryPullPush(aGraph, vertex, secondVertex, connectionIndex)) {
						aPushSuccessfulFlag.set_device();
					}
				}
			}
		}
	}
}


template<typename TFlow>
CUGIP_GLOBAL void
pushThroughTLinksFromSourceKernel(GraphCutData<TFlow> aGraph)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		float capacity = aGraph.sourceTLinkCapacity(index);
		if (capacity > 0.0) {
			aGraph.excess(index) += capacity;
		}
	}
}

template<typename TFlow>
CUGIP_GLOBAL void
pushThroughTLinksToSinkKernel(GraphCutData<TFlow> aGraph, TFlow *aTLinks)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount() && aGraph.sinkTLinkCapacity(index) > 0.0f) {
		float excess = aGraph.excess(index);
		if (excess > 0.0f) {
			aGraph.excess(index) -= excess;
			aTLinks[index] += excess;
		}
		// TODO handle situation when sink capacity isn't enough
	}
}


template<typename TFlow>
CUGIP_GLOBAL void
initResidualsKernel(TFlow *aWeightsForward, TFlow *aWeightsBackward, EdgeResidualsRecord<TFlow> *aResiduals, int aSize)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if (edgeIdx < aSize) {
		aResiduals[edgeIdx].residuals[0] = aWeightsForward[edgeIdx];
		aResiduals[edgeIdx].residuals[1] = aWeightsBackward[edgeIdx];
	}
}

template<typename TFlow, int tBlockSize>
CUGIP_DECL_DEVICE void
gatherScan(
	GraphCutData<TFlow> &aGraph,
	ParallelQueueView<int> aVertices,
	int aStartIndex,
	int aLevelEnd,
	int tid,
	int aCurrentLevel)
{
	__shared__ int buffer[tBlockSize+1];
	__shared__ int vertices[tBlockSize+1];
	//__shared__ int storeIndices[tBlockSize];
	//__shared__ int currentQueueRunStart;
	int vertexId = -1;
	int neighborCount = 0;
	// neighbor starting index (r)
	int index = 0;
	if (aStartIndex + tid < aLevelEnd) {
		vertexId = aVertices.get_device(aStartIndex + tid);
		neighborCount = aGraph.neighborCount(vertexId);
		index = aGraph.firstNeighborIndex(vertexId);
	}
	int neighborEnd = index + neighborCount;
	int rsvRank = block_prefix_sum(tid, tBlockSize, neighborCount, buffer);
	__syncthreads();
	int total = buffer[tBlockSize];
	int ctaProgress = 0;
	int remain = 0;
	while ((remain = total - ctaProgress) > 0) {
		while ((rsvRank < ctaProgress + tBlockSize) && index < neighborEnd) {
			buffer[rsvRank - ctaProgress] = index;
			vertices[rsvRank - ctaProgress] = vertexId;
			++rsvRank;
			++index;
		}
		__syncthreads();
		int shouldAppend = 0;
		int secondVertex = -1;
		if (tid < min(remain, tBlockSize)) {
			int firstVertex = vertices[tid];
			secondVertex = aGraph.secondVertex(buffer[tid]);
			int label = aGraph.label(secondVertex);
			TFlow residual = aGraph.residuals(aGraph.connectionIndex(buffer[tid])).getResidual(firstVertex > secondVertex);
			if (label == INVALID_LABEL && residual > 0.0f) {
				shouldAppend = (INVALID_LABEL == atomicCAS(&(aGraph.label(secondVertex)), INVALID_LABEL, aCurrentLevel)) ? 1 : 0;
			}
		}
		ctaProgress += tBlockSize;
		__syncthreads();
		if (shouldAppend) {
			//printf("%d\n", secondVertex);
			aVertices.append(secondVertex);
		}
		/*int queueOffset = block_prefix_sum(tid, tBlockSize, shouldAppend, buffer);
		if (tid == 0) {
			int totalAdded = buffer[tBlockSize - 1];
			currentQueueRunStart = aVertices.allocate(totalAdded);
		}
		__syncthreads();
		//TODO - store in shared buffer and then in global memory
		if (shouldAppend) {
			aVertices[currentQueueRunStart + queueOffset];
		}*/
		__syncthreads();
		//break;
	}
}

template<typename TFlow>
CUGIP_GLOBAL void
bfsPropagationKernel2(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		GraphCutData<TFlow> aGraph,
		int aCurrentLevel)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x;// + threadIdx.x;

	gatherScan<TFlow, 512>(
		aGraph,
		aVertices,
		index,
		aStart + aCount,
		threadIdx.x,
		aCurrentLevel);
}

CUGIP_GLOBAL void
testBlockScan()
{
	__shared__ int buffer[512+1];
	int rsvRank = block_prefix_sum(threadIdx.x, 512, 2, buffer);
	__syncthreads();
	printf("%d - %d - %d\n", threadIdx.x, rsvRank, buffer[threadIdx.x]);
	__syncthreads();
	if(!threadIdx.x) printf("%d - %d - %d\n", 512, -1, buffer[512]);
}

} // namespace cugip

