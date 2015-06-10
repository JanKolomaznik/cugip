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

template<typename TGraph>
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
};

struct WorkLimits
{
	/*CUGIP_DECL_DEVICE
	WorkLimits()
	{
		assert(false);
	}*/
	int elements;
	int offset;
	int guardedOffset;
	int guardedElements;
	int outOfBounds;
};

struct WorkDistribution
{
	int start;
	int count;
	int gridSize;

	int totalGrains;
	int grainsPerBlock;
	int extraGrains;

	WorkDistribution(int aStart, int aCount, int aGridSize, int aScheduleGranularity)
		: start(aStart)
		, count(aCount)
		, gridSize(aGridSize)
	{
		totalGrains = (count + aScheduleGranularity -1) / aScheduleGranularity;
		grainsPerBlock = totalGrains / gridSize;
		extraGrains = totalGrains - (grainsPerBlock * gridSize);
	}

	template<int tTileSize, int tScheduleGranularity>
	CUGIP_DECL_DEVICE WorkLimits
	getWorkLimits() const
	{
		// TODO TILE_SIZE ?
		WorkLimits workLimits;

		if (blockIdx.x < extraGrains) {
			// This CTA gets grains_per_cta+1 grains
			workLimits.elements = (grainsPerBlock + 1) * tScheduleGranularity;
			workLimits.offset = start + workLimits.elements * blockIdx.x;
		} else if (blockIdx.x < totalGrains) {
			// This CTA gets grains_per_cta grains
			workLimits.elements = grainsPerBlock * tScheduleGranularity;
			workLimits.offset = start + (workLimits.elements * blockIdx.x) + (extraGrains * tScheduleGranularity);
		} else {
			// This CTA gets no work (problem small enough that some CTAs don't even a single grain)
			workLimits.elements = 0;
			workLimits.offset = 0;
		}

		// The offset at which this CTA is out-of-bounds
		workLimits.outOfBounds = workLimits.offset + workLimits.elements;

		// Correct for the case where the last CTA having work has rounded its last grain up past the end
		if (/*workLimits.last_block = */workLimits.outOfBounds >= count) {
			workLimits.outOfBounds = start + count;
			workLimits.elements = count - workLimits.offset - start;
		}

		// The number of extra guarded-load elements to process afterward (always
		// less than a full tile)
		workLimits.guardedElements = workLimits.elements & (tTileSize - 1);

		// The tile-aligned limit for full-tile processing
		workLimits.guardedOffset = workLimits.outOfBounds - workLimits.guardedElements;


		/*int threadCount = blockDim.x * gridDim.x;
		//int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		//int index = blockId * blockDim.x;

		int itemsPerThread = (count + threadCount - 1) / threadCount;
		int itemsPerBlock = itemsPerThread * blockDim.x;
		int relativeOffset = blockIdx.x * itemsPerBlock;

		workLimits.offset = start + relativeOffset;
		if ((relativeOffset + itemsPerBlock) <= count) {
			workLimits.guardedOffset = start + relativeOffset + itemsPerBlock;
			workLimits.elements = itemsPerBlock;
		} else {
			workLimits.elements = count % itemsPerThread;
			workLimits.guardedElements = workLimits.elements % blockDim.x;
			workLimits.guardedOffset = start + workLimits.elements - workLimits.guardedElements;
		}*/
		return workLimits;
	}
};

template<typename TGraph, typename TPolicy>
struct TileProcessor
{
	TGraph &mGraph;
	ParallelQueueView<int> mVertices;
	// shared memory temporary buffer
	int *offsetScratch;

	CUGIP_DECL_DEVICE
	TileProcessor(
		TGraph &aGraph,
		ParallelQueueView<int> &aVertices
	)
		: mGraph(aGraph)
		, mVertices(aVertices)
	{
		assert(false && "offsetScratch not initialized");
	}

	/*void
	processTile(int offset);*/

	CUGIP_DECL_DEVICE void
	processTile(int aOffset, int aCount)
	{
		typedef cub::BlockScan<int, TPolicy::THREADS> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;

		Tile<TGraph> tile;
		tile.load(mGraph, aOffset, aCount);

		tile.getAdjacencyList(mGraph);

		int total = 0;
		int current = 0;
		BlockScan(temp_storage).ExclusiveSum(tile.listLength, current, total);
		__syncthreads();

		tile.progress = 0;
		while (tile.progress < total) {
			expand(tile);
			__syncthreads();

			int scratchRemainder = min<int>(TPolicy::SCRATCH_ELEMENTS, total - tile.progress);
			for (int scratchOffset = 0; scratchOffset < scratchRemainder; scratchOffset += TPolicy::THREADS) {
				int neighborId = -1;
				int shouldAdd = 0;
				if (scratchOffset + threadIdx.x < scratchRemainder) {
					neighborId = cullNeighbors(scratchOffset + threadIdx.x);
					shouldAdd = neighborId != -1 ? 1 : 0;
				}

				int newQueueOffset = 0;
				int totalItems = 0;
				BlockScan(temp_storage).ExclusiveSum(shouldAdd, newQueueOffset, totalItems);
				int currentQueueRunStart = mVertices.allocate(totalItems);
				if (neighborId != -1) {
					mVertices.get_device(currentQueueRunStart + newQueueOffset) = neighborId;
				}
			}

			tile.progress += TPolicy::THREADS;
			__syncthreads();
		}
	}

	CUGIP_DECL_DEVICE int
	cullNeighbors(int scratchIndex) {
		int neighborId = mGraph.secondVertex(offsetScratch[scratchIndex]);
		int connectionId = mGraph.connectionIndex(offsetScratch[scratchIndex]);
		auto residuals = mGraph.residuals(connectionId);

		return neighborId;
	}

	CUGIP_DECL_DEVICE void
	expand(Tile<TGraph> &aTile)
	{
		int scratchOffset = aTile.listStart + aTile.listProgress - aTile.progress; // ??
		while (aTile.listProgress < aTile.listLength && scratchOffset < TPolicy::SCRATCH_ELEMENTS) {
			offsetScratch[scratchOffset] = aTile.listStart + aTile.listProgress;
			++aTile.listProgress;
			++scratchOffset;
		}
	}
};

template<typename TGraph, typename TPolicy>
struct SweepPass
{
	static CUGIP_DECL_DEVICE void
	invoke(
		TGraph &aGraph,
		ParallelQueueView<int> &aVertices,
		const WorkDistribution &aWorkDistribution,
		int aCurrentLevel)
	{
		WorkLimits workLimits = aWorkDistribution.template getWorkLimits<TPolicy::TILE_SIZE, TPolicy::SCHEDULE_GRANULARITY>();

		if (workLimits.elements == 0) {
			return;
		}

		TileProcessor<TGraph, TPolicy> tileProcessor(aGraph, aVertices);
		while (workLimits.offset < workLimits.guardedOffset) {
			tileProcessor.processTile(
					workLimits.offset,
					TPolicy::TILE_SIZE);
			workLimits.offset += TPolicy::TILE_SIZE;
		}

		if (workLimits.guardedElements) {
			tileProcessor.processTile(
					workLimits.guardedOffset,
					workLimits.guardedElements);
		}
	}
};


template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
bfsPropagationKernel_b40c(
		ParallelQueueView<int> aVertices,
		WorkDistribution aWorkDistribution,
		TGraph aGraph,
		int aCurrentLevel)
{
	/*uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x;// + threadIdx.x;*/

	/*if (threadIdx.x == 0) {

	}*/
	/*gatherScan<TGraph, TPolicy>(
		aGraph,
		aVertices,
		aStart + index,
		aStart + aCount,
		aCurrentLevel);*/
	SweepPass<TGraph, TPolicy>::invoke(
			aGraph,
			aVertices,
			aWorkDistribution,
			aCurrentLevel);
}
#if 0
template<typename TGraph, typename TPolicy>
CUGIP_DECL_DEVICE void
gatherScan(
	TGraph &aGraph,
	ParallelQueueView<int> aVertices,
	int aStartIndex,
	int aLevelEnd,
	int aCurrentLevel)
{
	typedef cub::BlockScan<int, TPolicy::THREADS> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;

	__shared__ int buffer[TPolicy::THREADS+1];
	__shared__ int vertices[TPolicy::THREADS+1];
	//__shared__ int storeIndices[tBlockSize];
	__shared__ int currentQueueRunStart;
	Tile<TGraph> tile;
	tile.vertexId = -1;
	tile.listLength = 0;
	// neighbor starting index (r)
	tile.listStart = 0;
	tile.fill();
	if (aStartIndex + threadIdx.x < aLevelEnd) {
		tile.vertexId = aVertices.get_device(aStartIndex + threadIdx.x);
		assert(tile.vertexId >= 0);
		neighborCount = aGraph.neighborCount(tile.vertexId);
		index = aGraph.firstNeighborIndex(tile.vertexId);
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
		while ((rsvRank < ctaProgress + TPolicy::THREADS) && index < neighborEnd) {
			buffer[rsvRank - ctaProgress] = index;
			vertices[rsvRank - ctaProgress] = vertexId;
			++rsvRank;
			++index;
		}
		__syncthreads();
		int shouldAppend = 0;
		int secondVertex = -1;
		if (threadIdx.x < min<int>(remain, TPolicy::THREADS)) {
			int firstVertex = vertices[threadIdx.x];

			secondVertex = aGraph.secondVertex(buffer[threadIdx.x]);
			int label = aGraph.label(secondVertex);
			auto residual = aGraph.residuals(aGraph.connectionIndex(buffer[threadIdx.x])).getResidual(firstVertex > secondVertex);
			if (label == TPolicy::INVALID_LABEL && residual > 0.0f) {
				shouldAppend = (TPolicy::INVALID_LABEL == atomicCAS(&(aGraph.label(secondVertex)), TPolicy::INVALID_LABEL, aCurrentLevel)) ? 1 : 0;
			}
		}
		__syncthreads();
		ctaProgress += TPolicy::THREADS;

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
#endif

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
#if 0
	static bool
	bfs_iteration2(TGraphData &aGraph, size_t &aCurrentLevel, std::vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue)
	{
		size_t level = aCurrentLevel;
		dim3 blockSize1D(TPolicy::THREADS, 1, 1);
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
#endif
	static bool
	bfs_iteration(TGraphData &aGraph, size_t &aCurrentLevel, std::vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue)
	{
		//size_t level = aCurrentLevel;
		dim3 blockSize1D(TPolicy::THREADS, 1, 1);
		int frontierSize = aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1];
		dim3 levelGridSize1D(1 + (frontierSize - 1) / (blockSize1D.x), 1, 1);
		CUGIP_CHECK_ERROR_STATE("Before bfsPropagationKernel_b40c()");

		bfsPropagationKernel_b40c<TGraphData, TPolicy><<<levelGridSize1D, blockSize1D>>>(
			aVertexQueue,
			WorkDistribution(aLevelStarts[aCurrentLevel - 1], frontierSize, levelGridSize1D.x, TPolicy::SCHEDULE_GRANULARITY),
			aGraph,
			aCurrentLevel + 1);
		++aCurrentLevel;
		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel_b40c()");
		int lastLevelSize = aVertexQueue.size();
		//CUGIP_DPRINT("LastLevelSize " << lastLevelSize);
		if (lastLevelSize == aLevelStarts.back()) {
			return true;
		}
		//CUGIP_DPRINT("Level " << (aCurrentLevel + 1) << " size: " << (lastLevelSize - aLevelStarts.back()));
		//if (currentLevel == 2) break;
		aLevelStarts.push_back(lastLevelSize);
		return false;
	}

};

} // namespace cugip
