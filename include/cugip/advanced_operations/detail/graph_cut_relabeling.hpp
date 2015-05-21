#pragma once

namespace cugip {

/*
template<typename TFlow>
void
Graph<TFlow>::assign_label_by_distance()
{
	//CUGIP_DPRINT("assign_label_by_distance");
	dim3 blockSize1D(512, 1, 1);
	dim3 gridSize1D((mGraphData.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x), 1);

	mVertexQueue.clear();
	initBFSKernel<<<gridSize1D, blockSize1D>>>(mVertexQueue.view(), mGraphData);

	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After initBFSKernel()");
	int lastLevelSize = mVertexQueue.size();
	//CUGIP_DPRINT("Level 1 size: " << lastLevelSize);
	mLevelStarts.clear();
	mLevelStarts.push_back(0);
	mLevelStarts.push_back(lastLevelSize);
	size_t currentLevel = 1;
	bool finished = lastLevelSize == 0;
	while (!finished) {
		finished = bfs_iteration(currentLevel);
	}

	cudaThreadSynchronize();
	CUGIP_DPRINT("Active vertex count = " << mLevelStarts.back());
	CUGIP_CHECK_ERROR_STATE("After assign_label_by_distance()");
}*/




template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
initBFSKernel(ParallelQueueView<int> aVertices, TGraph aGraph)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		int label = TPolicy::INVALID_LABEL;
			//printf("checking %d - %d\n", index, aGraph.vertexCount());
		if (aGraph.sinkTLinkCapacity(index) > 0.0) {
			label = 1;
			aVertices.append(index);
			//printf("adding %d\n", index);
		}
		aGraph.label(index) = label;
	}
}

template<typename TFlow, typename TPolicy>
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
			int shouldAppend = 0;
			TFlow residual = aGraph.residuals(aGraph.connectionIndex(firstNeighborIndex + i)).getResidual(vertex > secondVertex);
			if (label == TPolicy::INVALID_LABEL && residual > 0.0f) {
				aGraph.label(secondVertex) = aCurrentLevel; //TODO atomic
				aVertices.append(secondVertex);
				shouldAppend = 1;
				//printf("%d\n", secondVertex);
			}
			printf("SV_%d %d -> %d : %d, %f %d\n", aCurrentLevel, vertex, secondVertex, label, residual, shouldAppend);
			//printf("%d, %d, %d\n", vertex, secondVertex, label);
		}
		//printf("%d, %d\n", vertex, neighborCount);
		//printf("%d %d %d\n", index, blockId, aCount);
	}
}



template<typename TGraph, int tBlockSize, typename TPolicy>
CUGIP_DECL_DEVICE void
gatherScan(
	TGraph &aGraph,
	ParallelQueueView<int> aVertices,
	int aStartIndex,
	int aLevelEnd,
	int tid,
	int aCurrentLevel)
{
	__shared__ int buffer[tBlockSize+1];
	__shared__ int vertices[tBlockSize+1];
	//__shared__ int storeIndices[tBlockSize];
	__shared__ int currentQueueRunStart;
	int vertexId = -1;
	int neighborCount = 0;
	// neighbor starting index (r)
	int index = 0;
	if (aStartIndex + tid < aLevelEnd) {
		vertexId = aVertices.get_device(aStartIndex + tid);
		assert(vertexId >= 0);
		neighborCount = aGraph.neighborCount(vertexId);
		index = aGraph.firstNeighborIndex(vertexId);
	}//printf("%d index %d\n", aCurrentLevel, aStartIndex + tid);
	int neighborEnd = index + neighborCount;
	ScanResult<int> prefix_sum = block_prefix_sum_ex<int>(tid, tBlockSize, neighborCount, buffer);
	//int rsvRank = block_prefix_sum_ex(tid, tBlockSize, neighborCount, buffer);
	int rsvRank = prefix_sum.current;
	int total = prefix_sum.total;
	__syncthreads();
	//int total = buffer[tBlockSize];
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
			auto residual = aGraph.residuals(aGraph.connectionIndex(buffer[tid])).getResidual(firstVertex > secondVertex);
			if (label == TPolicy::INVALID_LABEL && residual > 0.0f) {
				shouldAppend = (TPolicy::INVALID_LABEL == atomicCAS(&(aGraph.label(secondVertex)), TPolicy::INVALID_LABEL, aCurrentLevel)) ? 1 : 0;
			}
		}
		__syncthreads();
		ctaProgress += tBlockSize;
		//int queueOffset = block_prefix_sum_ex(tid, tBlockSize, shouldAppend, buffer);
		ScanResult<int> queueOffset = block_prefix_sum_ex<int>(tid, tBlockSize, shouldAppend, buffer);
		__syncthreads();
		if (tid == 0) {
			currentQueueRunStart = aVertices.allocate(queueOffset.total);
		}
		__syncthreads();
		//TODO - store in shared buffer and then in global memory
		if (shouldAppend) {
			aVertices.get_device(currentQueueRunStart + queueOffset.current) = secondVertex;
		}
		__syncthreads();
	}
}

template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
bfsPropagationKernel2(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		TGraph aGraph,
		int aCurrentLevel)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x;// + threadIdx.x;

	gatherScan<TGraph, 512, TPolicy>(
		aGraph,
		aVertices,
		aStart + index,
		aStart + aCount,
		threadIdx.x,
		aCurrentLevel);
}

template<typename TFlow>
CUGIP_GLOBAL void
bfsPropagationKernel3(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		GraphCutData<TFlow> aGraph,
		int aCurrentLevel,
		ParallelQueueView<int> aLevelStarts)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x;// + threadIdx.x;
	int m = 0;
	do {
		__syncthreads();
		gatherScan<TFlow, 512>(
			aGraph,
			aVertices,
			aStart + index,
			aStart + aCount,
			threadIdx.x,
			aCurrentLevel);
		__syncthreads();
		int size = aVertices.device_size();
		aStart += aCount;
		aCount = size - aStart;
		if (threadIdx.x == 0) {
			aLevelStarts.append(size);
		}
		++aCurrentLevel;
		++m;
	} while (m < 100 && aCount <= 512 && aCount > 0);//(false);
}

template<typename TGraphData, typename TPolicy>
struct Relabel
{
	static void
	compute(
		TGraphData &aGraph,
		ParallelQueueView<int> &aVertexQueue,
		std::vector<int> &aLevelStarts)
	{
		dim3 blockSize1D(512, 1, 1);
		dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x), 1);

		aVertexQueue.clear();
		initBFSKernel<TGraphData, TPolicy><<<gridSize1D, blockSize1D>>>(aVertexQueue, aGraph);

		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		CUGIP_CHECK_ERROR_STATE("After initBFSKernel()");
		int lastLevelSize = aVertexQueue.size();
		//CUGIP_DPRINT("Level 1 size: " << lastLevelSize);
		aLevelStarts.clear();
		aLevelStarts.push_back(0);
		aLevelStarts.push_back(lastLevelSize);
		size_t currentLevel = 1;
		bool finished = lastLevelSize == 0;
		while (!finished) {
			finished = bfs_iteration(aGraph, currentLevel, aLevelStarts, aVertexQueue);
		}

		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		//CUGIP_DPRINT("Active vertex count = " << aLevelStarts.back());
		CUGIP_CHECK_ERROR_STATE("After assign_label_by_distance()");
	}

	static bool
	bfs_iteration(TGraphData &aGraph, size_t &aCurrentLevel, std::vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue)
	{
		size_t level = aCurrentLevel;
		dim3 blockSize1D(512, 1, 1);
		int frontierSize = aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1];
		dim3 levelGridSize1D(1 + (frontierSize - 1) / (blockSize1D.x), 1, 1);
		CUGIP_CHECK_ERROR_STATE("Before bfsPropagationKernel()");
		/*if (frontierSize <= blockSize1D.x) {
			mLevelStartsQueue.clear();
			bfsPropagationKernel3<<<levelGridSize1D, blockSize1D>>>(
				mVertexQueue.view(),
				aLevelStarts[aCurrentLevel - 1],
				frontierSize,
				mGraphData,
				aCurrentLevel + 1,
				mLevelStartsQueue.view());
			cudaThreadSynchronize();
			CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel3)");
			thrust::host_vector<int> starts;
			mLevelStartsQueue.fill_host(starts);
			int originalStart = aLevelStarts.back();
			int lastStart = originalStart;
			for (int i = 0; i < starts.size(); ++i) {
				if (starts[i] == lastStart) {
					lastStart = -1;
				} else {
					lastStart = starts[i];
				}
				aLevelStarts.push_back(starts[i]);
			}
			aCurrentLevel = aLevelStarts.size() - 1;
			//CUGIP_DPRINT("Level bundle " << (level + 1) << "-" << (aCurrentLevel + 1) << " size: " << (originalStart - aLevelStarts.back()));
			return (lastStart == originalStart) || (lastStart == -1);
		} else*/ {
			bfsPropagationKernel2<TGraphData, TPolicy><<<levelGridSize1D, blockSize1D>>>(
				aVertexQueue,
				aLevelStarts[aCurrentLevel - 1],
				frontierSize,
				aGraph,
				aCurrentLevel + 1);
			++aCurrentLevel;
			cudaThreadSynchronize();
			CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel2()");
			int lastLevelSize = aVertexQueue.size();
			//CUGIP_DPRINT("LastLevelSize " << lastLevelSize);
			if (lastLevelSize == aLevelStarts.back()) {
				return true;
			}
			//CUGIP_DPRINT("Level " << (aCurrentLevel + 1) << " size: " << (lastLevelSize - aLevelStarts.back()));
			//if (currentLevel == 2) break;
			aLevelStarts.push_back(lastLevelSize);
		}

		return false;
	}

};

} // namespace cugip
