#pragma once

#include <cub/block/block_scan.cuh>

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

struct Tile {
	int vertexId;
	int listStart;
	int listLength;

	int listProgress;
	int progress;

	CUGIP_DECL_DEVICE void
	load(TGraph &aGraph, int aOffset, int aCount)
	{
		assert(false);
	}

	CUGIP_DECL_DEVICE void
	getAdjacencyList(TGraph &aGraph)
	{
		if (vertexId >= 0) {
			listStart = aGraph.firstNeighborIndex(vertexId);
			listLength = aGraph.firstNeighborIndex(vertexId + 1) - listStart;
		}
	}

	CUGIP_DECL_DEVICE void
	expand(TileProcessor &aProcessor)
	{
		int scratchOffset = listStart + listProgress - progress; // ??
		while (listProgress < listLength && scratchOffset < TileProcessor::OFFSET_ELEMENTS) {
			aProcessor.offsetScratch[scratchOffset] = listStart + listProgress;
			++listProgress;
			++scratchOffset;
		}
	}
};

struct WorkLimits
{
	CUGIP_DECL_DEVICE
	WorkLimits()
	{
		assert(false);
	}
	int elements;
	int offset;
	int guardedOffset;
	int guardedElements;
};

struct TileProcessor
{
	CUGIP_DECL_DEVICE
	TileProcessor()
	{}

	/*void
	processTile(int offset);*/

	CUGIP_DECL_DEVICE void
	processTile(int aOffset, int aCount)
	{
		typedef cub::BlockScan<int, TPolicy::BLOCK_SIZE> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;

		Tile tile;
		tile.load(mGraph, aOffset, aCount);

		tile.getAdjacencyList(mGraph);

		int total = 0;
		int current = 0;
		BlockScan(temp_storage).ExclusiveSum(tile.listLength, current, total);
		__syncthreads();

		tile.progress = 0;
		while (progress < total) {
			tile.expand(*this);
			__syncthreads();

			int scratchRemainder = min<int>(TPolicy::BLOCK_SIZE, total - tile.progress);
			for (int scratchOffset = 0; scratchOffset < scratchRemainder; scratchOffset += TPolicy::THREADS) {
				int neighborId = -1;

				if (scratchOffset + threadIdx.x < scratchRemainder) {
					neighbor_id = mGraph.
				}
			}

			progress += TPolicy::BLOCK_SIZE;
			__syncthreads();
		}
	}
};

template<typename TPolicy>
CUGIP_DECL_DEVICE void
sweepPass()
{
	WorkLimits workLimits;

	if (workLimits.elements == 0) {
		return;
	}

	typename TPolicy::TileProcessor tileProcessor;
	while (workLimits.offset < workLimits.guardedOffset) {
		tileProcessor.ProcessTile(workLimits.offset);
		workLimits.offset += TPolicy::TILE_SIZE;
	}

	if (workLimits.guardedElements) {
		tileProcessor.ProcessTile(
				workLimits.guardedOffset,
				workLimits.guardedElements);
	}
}

template<typename TGraph, typename TPolicy>
CUGIP_DECL_DEVICE void
gatherScan(
	TGraph &aGraph,
	ParallelQueueView<int> aVertices,
	int aStartIndex,
	int aLevelEnd,
	int aCurrentLevel)
{
	typedef cub::BlockScan<int, TPolicy::BLOCK_SIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;

	__shared__ int buffer[TPolicy::BLOCK_SIZE+1];
	__shared__ int vertices[TPolicy::BLOCK_SIZE+1];
	//__shared__ int storeIndices[tBlockSize];
	__shared__ int currentQueueRunStart;
	Tile tile;
	tile.vertexId = -1;
	tile.listLength = 0;
	// neighbor starting index (r)
	tile.listStart = 0;
	tile.fill()
	if (aStartIndex + threadIdx.x < aLevelEnd) {
		vertexId = aVertices.get_device(aStartIndex + threadIdx.x);
		assert(vertexId >= 0);
		neighborCount = aGraph.neighborCount(vertexId);
		index = aGraph.firstNeighborIndex(vertexId);
	}//printf("%d index %d\n", aCurrentLevel, aStartIndex + threadIdx.x);
	int neighborEnd = index + neighborCount;
	int rsvRank = 0;
	int total = 0;
	BlockScan(temp_storage).ExclusiveSum(neighborCount, rsvRank, total);
	//ScanResult<int> prefix_sum = block_prefix_sum_ex<int>(threadIdx.x, tBlockSize, neighborCount, buffer);
	//int rsvRank = block_prefix_sum_ex(threadIdx.x, tBlockSize, neighborCount, buffer);
	//int rsvRank = prefix_sum.current;
	//int total = prefix_sum.total;
	__syncthreads();
	//int total = buffer[tBlockSize];
	int ctaProgress = 0;
	int remain = 0;
	while ((remain = total - ctaProgress) > 0) {
		while ((rsvRank < ctaProgress + TPolicy::BLOCK_SIZE) && index < neighborEnd) {
			buffer[rsvRank - ctaProgress] = index;
			vertices[rsvRank - ctaProgress] = vertexId;
			++rsvRank;
			++index;
		}
		__syncthreads();
		int shouldAppend = 0;
		int secondVertex = -1;
		if (threadIdx.x < min<int>(remain, TPolicy::BLOCK_SIZE)) {
			int firstVertex = vertices[threadIdx.x];

			secondVertex = aGraph.secondVertex(buffer[threadIdx.x]);
			int label = aGraph.label(secondVertex);
			auto residual = aGraph.residuals(aGraph.connectionIndex(buffer[threadIdx.x])).getResidual(firstVertex > secondVertex);
			if (label == TPolicy::INVALID_LABEL && residual > 0.0f) {
				shouldAppend = (TPolicy::INVALID_LABEL == atomicCAS(&(aGraph.label(secondVertex)), TPolicy::INVALID_LABEL, aCurrentLevel)) ? 1 : 0;
			}
		}
		__syncthreads();
		ctaProgress += TPolicy::BLOCK_SIZE;

		int totalOffset = 0;
		int itemOffset = 0;
		BlockScan(temp_storage).ExclusiveSum(shouldAppend, itemOffset, totalOffset);
		//int queueOffset = block_prefix_sum_ex(threadIdx.x, tBlockSize, shouldAppend, buffer);
		//ScanResult<int> queueOffset = block_prefix_sum_ex<int>(threadIdx.x, tBlockSize, shouldAppend, buffer);
		__syncthreads();
		if (threadIdx.x == 0) {
			currentQueueRunStart = aVertices.allocate(/*queueOffset.total*/totalOffset);
		}
		__syncthreads();
		//TODO - store in shared buffer and then in global memory
		if (shouldAppend) {
			aVertices.get_device(currentQueueRunStart + itemOffset/*queueOffset.current*/) = secondVertex;
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

	gatherScan<TGraph, TPolicy>(
		aGraph,
		aVertices,
		aStart + index,
		aStart + aCount,
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
		dim3 blockSize1D(TPolicy::BLOCK_SIZE, 1, 1);
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
