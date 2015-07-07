#pragma once

#include <cugip/parallel_queue.hpp>

namespace cugip {


template<typename TFlow>
CUGIP_GLOBAL void
pushThroughTLinksFromSourceKernel(GraphCutData<TFlow> aGraph)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		float capacity = aGraph.sourceTLinkCapacity(index);
		//printf("psss %d -> %f\n", index, capacity);
		if (capacity > 0.0) {
			aGraph.excess(index) += capacity;
		}
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
tryPullPush(GraphCutData<TFlow> &aGraph, int aFrom, int aTo, int aConnectionIndex)
{
	EdgeResidualsRecord<TFlow> &edge = aGraph.residuals(aConnectionIndex);
	TFlow residual = edge.getResidual(aFrom < aTo);
	TFlow flow = tryPull(aGraph, aFrom, aTo, residual);
	if (flow > 0.0f) {
		//printf("try pull push %d %d -> %d %d %f\n", aGraph.label(aFrom), aFrom, aGraph.label(aTo), aTo, flow);
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
CUGIP_DECL_DEVICE void
pushImplementation(
		GraphCutData<TFlow> &aGraph,
		ParallelQueueView<int> &aVertices,
		int index,
		int aLevelEnd,
		device_flag_view &aPushSuccessfulFlag)
{
	int vertex = aVertices.get_device(index);
	//printf("p %d - %f\n", vertex, aGraph.excess(vertex));
	if (aGraph.excess(vertex) > 0.0f) {
		int neighborCount = aGraph.neighborCount(vertex);
		int firstNeighborIndex = aGraph.firstNeighborIndex(vertex);
		int label = aGraph.label(vertex);
		for (int i = 0; i < neighborCount; ++i) {
			int secondVertex = aGraph.secondVertex(firstNeighborIndex + i);
			int secondLabel = aGraph.label(secondVertex);
			if (label > secondLabel && secondLabel >= 0) {
				int connectionIndex = aGraph.connectionIndex(firstNeighborIndex + i);
				if (tryPullPush(aGraph, vertex, secondVertex, connectionIndex)) {
					printf("proc %d %d -> %d %d\n", vertex, label, secondVertex, secondLabel);
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
		device_flag_view aPushSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = aLevelStart + blockId * blockDim.x + threadIdx.x;

	if (index < aLevelEnd) {
		pushImplementation(aGraph, aVertices, index, aLevelEnd, aPushSuccessfulFlag);
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
			pushImplementation(aGraph, aVertices, index, aLevelStarts[level], aPushSuccessfulFlag);
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
	static bool
	compute(
		TGraphData &aGraph,
		ParallelQueueView<int> &aVertexQueue,
		std::vector<int> &aLevelStarts)
	{
		thrust::host_vector<int> starts;
		starts.reserve(1000);
		thrust::device_vector<int> device_starts;
		device_starts.reserve(1000);
		//CUGIP_DPRINT("push()");
		dim3 blockSize1D(512);
		device_flag pushSuccessfulFlag;

		for (int i = aLevelStarts.size() - 1; i > 0; --i) {
			int count = aLevelStarts[i] - aLevelStarts[i-1];
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
						aLevelStarts[i-1],
						aLevelStarts[i],
						pushSuccessfulFlag.view());
			}
			CUGIP_CHECK_RESULT(cudaThreadSynchronize());
			//CUGIP_DPRINT("-------------------------------");
		}
		CUGIP_CHECK_ERROR_STATE("After push()");
		return pushSuccessfulFlag.check_host();
	}

};


} // namespace cugip
