#pragma once

#include <cugip/parallel_queue.hpp>

namespace cugip {


template<typename TFlow>
CUGIP_GLOBAL void
pushThroughTLinksFromSourceKernel(GraphCutData<TFlow> aGraph)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	while (index < aGraph.vertexCount()) {
		float capacity = aGraph.sourceTLinkCapacity(index);
		//printf("psss %d -> %f\n", index, capacity);
		if (capacity > 0.0) {
			aGraph.excess(index) += capacity;
		}
		index += blockDim.x * gridDim.x;
	}
}

template<typename TFlow>
CUGIP_GLOBAL void
pushThroughTLinksToSinkKernel(GraphCutData<TFlow> aGraph)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount() && aGraph.sinkTLinkCapacity(index) > 0.0f) {
		float excess = aGraph.excess(index);
		if (excess > 0.0f) {
			aGraph.excess(index) -= excess;
			aGraph.sinkFlow(index) += excess;
		}
		// TODO handle situation when sink capacity isn't enough
	}
}


/*
template<typename TFlow>
bool
Graph<TFlow>::push()
{
	thrust::host_vector<int> starts;
	starts.reserve(1000);
	thrust::device_vector<int> device_starts;
	device_starts.reserve(1000);
	//CUGIP_DPRINT("push()");
	dim3 blockSize1D(512);
	device_flag pushSuccessfulFlag;

	for (int i = mLevelStarts.size() - 2; i > 0; --i) {
		int count = mLevelStarts[i] - mLevelStarts[i-1];
		if (count <= blockSize1D.x) {
			starts.push_back(mLevelStarts[i]);
			while (i > 0 && (mLevelStarts[i] - mLevelStarts[i-1]) <= blockSize1D.x) {
				starts.push_back(mLevelStarts[i-1]);
				//CUGIP_DPRINT(i << " - " << mLevelStarts[i-1]);
				--i;
			}
			++i;
			//CUGIP_DPRINT("-------------------------------");
			device_starts = starts;
			dim3 gridSize1D(1);
			pushKernel2<<<gridSize1D, blockSize1D>>>(
					mGraphData,
					mVertexQueue.view(),
					thrust::raw_pointer_cast(device_starts.data()),
					device_starts.size(),
					pushSuccessfulFlag.view());

		} else {
			//CUGIP_DPRINT()
			dim3 gridSize1D((count + blockSize1D.x - 1) / (blockSize1D.x), 1);
			pushKernel<<<gridSize1D, blockSize1D>>>(
					mGraphData,
					mVertexQueue.view(),
					mLevelStarts[i-1],
					mLevelStarts[i],
					pushSuccessfulFlag.view());
		}
	cudaThreadSynchronize();
	//CUGIP_DPRINT("-------------------------------");
	}
	cudaThreadSynchronize();
	//CUGIP_DPRINT("-------------------------------");
	CUGIP_CHECK_ERROR_STATE("After push()");
	return pushSuccessfulFlag.check_host();
}*/

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
tryPullPush(GraphCutData<TFlow> &aGraph, int aFrom, int aTo, int aConnectionIndex, bool aConnectionSide)
{
	EdgeResidualsRecord<TFlow> &edge = aGraph.residuals(aConnectionIndex);
	TFlow residual = edge.getResidual(aConnectionSide);
	TFlow flow = tryPull(aGraph, aFrom, aTo, residual);
	if (flow > 0.0f) {
		//printf("try pull push %d %d -> %d %d %f - residual %f\n", aGraph.label(aFrom), aFrom, aGraph.label(aTo), aTo, flow, residual);
		atomicAdd(&(aGraph.excess(aTo)), flow);
		edge.getResidual(aConnectionSide) -= flow;
		edge.getResidual(!aConnectionSide) += flow;
		return true;
	}
	//printf("failed pull push %d %d -> %d %d %f - residual %f\n", aGraph.label(aFrom), aFrom, aGraph.label(aTo), aTo, flow, residual);
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
CUGIP_DECL_DEVICE void
pushImplementation(
		GraphCutData<TFlow> &aGraph,
		ParallelQueueView<int> &aVertices,
		int index,
		int aCurrentLevel,
		device_flag_view &aPushSuccessfulFlag)
{
	int vertex = aVertices.get_device(index);
	CUGIP_ASSERT(aCurrentLevel == aGraph.label(vertex));
	if (aGraph.excess(vertex) > 0.0f) {
		int neighborCount = aGraph.neighborCount(vertex);
		int firstNeighborIndex = aGraph.firstNeighborIndex(vertex);
		for (int i = 0; i < neighborCount; ++i) {
			int secondVertex = aGraph.secondVertex(firstNeighborIndex + i);
			int secondLabel = aGraph.label(secondVertex);
			if (aCurrentLevel > secondLabel && secondLabel >= 0) {
				int connectionIndex = aGraph.connectionIndex(firstNeighborIndex + i);
				bool connectionSide = aGraph.connectionSide(firstNeighborIndex + i);
				if (tryPullPush(aGraph, vertex, secondVertex, connectionIndex, connectionSide)) {
					aPushSuccessfulFlag.set_device();
				}
			}
		}
	}
}

/*
template<typename TFlow>
CUGIP_DECL_DEVICE void
pushImplementation(
		GraphCutData<TFlow> &aGraph,
		ParallelQueueView<int> &aVertices,
		int index,
		int aLevelEnd,
		device_flag_view &aPushSuccessfulFlag)
{
	__shared__ int buffer[513];
	__shared__ int vertices[512];
	int vertex = aVertices.get_device(index);

	int shouldProcess = (index < aLevelEnd && aGraph.excess(vertex) > 0.0f) ? 1 : 0;
	__syncthreads();
	auto offset = block_prefix_sum_ex(threadIdx.x, 512, shouldProcess, buffer);
	__syncthreads();
	//int total = buffer[512];
	__syncthreads();
	if (shouldProcess) {
		//if (offset.current > 511) printf("aaa %d - %d\n", offset.current, offset.total);
		vertices[offset.current] = vertex;
	}
	__syncthreads();
	//printf("%d %d - %d - %d/%d\n", blockIdx.x, threadIdx.x, shouldProcess, offset.current, offset.total);
	if (threadIdx.x < offset.total) {
		//assert(offset.total <= 512);
		vertex = vertices[threadIdx.x];
	//if (aGraph.excess(vertex) > 0.0f) {
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
	__syncthreads();
}*/


template<typename TFlow>
CUGIP_GLOBAL void
pushKernel(
		GraphCutData<TFlow> aGraph,
		ParallelQueueView<int> aVertices,
		int aLevelStart,
		int aLevelEnd,
		int aCurrentLevel,
		device_flag_view aPushSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = aLevelStart + blockId * blockDim.x + threadIdx.x;

	while (index < aLevelEnd) {
		pushImplementation(aGraph, aVertices, index, aCurrentLevel, aPushSuccessfulFlag);
		index += blockDim.x * gridDim.x;
	}
}


template<typename TFlow>
CUGIP_GLOBAL void
pushKernel2(
		GraphCutData<TFlow> aGraph,
		ParallelQueueView<int> aVertices,
		int *aLevelStarts,
		int aLevelCount,
		device_flag_view aPushSuccessfulFlag)
{
	for (int level = 0; level < aLevelCount - 1; ++level) {
		int index = aLevelStarts[level + 1] + threadIdx.x;
		if (index < aLevelStarts[level]) {
			//if (threadIdx.x == 0) printf("aaaa %d\n", index);
			//pushImplementation(aGraph, aVertices, index, aLevelStarts[level], aPushSuccessfulFlag);
		}
		__syncthreads();
	}

	/*if (index < aLevelEnd) {
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
	}*/
}


template<typename TGraphData, typename TPolicy>
struct Push
{
	template<int tImplementationId>
	struct PushIteration
	{
		static void compute(
			TGraphData &aGraph,
			ParallelQueueView<int> &aVertexQueue,
			int aLevelStart,
			int aLevelEnd,
			int aCurrentLevel,
			device_flag_view aPushSuccessfulFlag)
		{
			dim3 blockSize1D(512);
			int count = aLevelEnd - aLevelStart;
			if (count == 0) {
				return;
			}
			CUGIP_ASSERT(count > 0);
			/*if (count <= blockSize1D.x) {
				starts.push_back(aLevelStarts[i]);
				while (i > 0 && (aLevelStarts[i] - aLevelStarts[i-1]) <= blockSize1D.x) {
					starts.push_back(aLevelStarts[i-1]);
					//CUGIP_DPRINT(i << " - " << aLevelStarts[i-1]);
					--i;
				}
				++i;
				//CUGIP_DPRINT("-------------------------------");
				device_starts = starts;
				dim3 gridSize1D(1);
				pushKernel2<<<gridSize1D, blockSize1D>>>(
						aGraph,
						aVertexQueue,
						thrust::raw_pointer_cast(device_starts.data()),
						device_starts.size(),
						pushSuccessfulFlag.view());

			} else */{
				//CUGIP_DFORMAT("level %1%", i-1);
				dim3 gridSize1D((count + blockSize1D.x - 1) / (blockSize1D.x), 1);
				pushKernel<<<gridSize1D, blockSize1D>>>(
						aGraph,
						aVertexQueue,
						aLevelStart,
						aLevelEnd,
						aCurrentLevel,
						aPushSuccessfulFlag);
			}
			CUGIP_CHECK_RESULT(cudaThreadSynchronize());
			//CUGIP_DPRINT("-------------------------------");

		}
	};

	bool
	compute(
		TGraphData &aGraph,
		ParallelQueueView<int> &aVertexQueue,
		std::vector<int> &aLevelStarts)
	{
		/*thrust::host_vector<int> starts;
		starts.reserve(1000);
		thrust::device_vector<int> device_starts;
		device_starts.reserve(1000);*/
		//CUGIP_DPRINT("push()");
		//dim3 blockSize1D(512);
		pushSuccessfulFlag.reset_host();
		for (int i = aLevelStarts.size() - 1; i > 0; --i) {
			PushIteration<TPolicy::PUSH_ITERATION_ALGORITHM>::compute(
					aGraph,
					aVertexQueue,
					aLevelStarts[i-1],
					aLevelStarts[i],
					i,
					pushSuccessfulFlag.view());
		}
		CUGIP_CHECK_ERROR_STATE("After push()");
		return pushSuccessfulFlag.check_host();
	}
	device_flag pushSuccessfulFlag;
};


} // namespace cugip
