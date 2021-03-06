#pragma once

#include <cub/block/block_scan.cuh>
#include <cub/grid/grid_barrier.cuh>
#include <cugip/parallel_queue.hpp>
#include <cugip/advanced_operations/detail/graph_cut_data.hpp>
#include <algorithm>
#include <thrust/fill.h>

#include "graph_cut_policies.hpp"

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
	int currentLevel = 1;
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
	typedef cub::BlockScan<int, TPolicy::THREADS> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	__shared__ int allocated_list_start;

	int label = TPolicy::INVALID_LABEL;
	if (index < aGraph.vertexCount()) {
			//printf("checking %d - %d\n", index, aGraph.vertexCount());
		if (aGraph.tLinkCapacity<TPolicy::cStartTLinkType>(index) > 0.0) {
			label = 1;
			//aVertices.append(index);
			//printf("adding %d\n", index);
		}
		aGraph.label(index) = label;
	}
	__syncthreads();
	int total = 0;
	int current = 0;
	BlockScan(temp_storage).ExclusiveSum(label != TPolicy::INVALID_LABEL ? 1 : 0, current, total);
	if (threadIdx.x == 0 && total > 0) {
		//printf("Allocating %d\n", total);
		allocated_list_start = aVertices.allocate(total);
	}
	__syncthreads();
	if (label != TPolicy::INVALID_LABEL) {
		//printf("adding %d\n", index);
		aVertices.get_device(allocated_list_start + current) = index;
	}
}



template<typename TGraph, typename TPolicy>
CUGIP_DECL_DEVICE void
propagateFromVertex(int vertex, ParallelQueueView<int> &aVertices, TGraph &aGraph, int aCurrentLevel, const TPolicy &aPolicy)
{
	//typedef cub::BlockScan<int, TPolicy::THREADS> BlockScan;
	//__shared__ typename BlockScan::TempStorage temp_storage;

	int neighborCount = aGraph.neighborCount(vertex);
	int firstNeighborIndex = aGraph.firstNeighborIndex(vertex);
	for (int i = firstNeighborIndex; i < firstNeighborIndex + neighborCount; ++i) {
		int secondVertex = aGraph.secondVertex(i);
		int label = aGraph.label(secondVertex);

		int connectionId = aGraph.connectionIndex(i);
		bool connectionSide = aGraph.connectionSide(i);
		auto residuals = aGraph.residuals(connectionId);
		bool isOpen = label == TPolicy::INVALID_LABEL && aPolicy.edgeTraversalCheck.invoke(connectionSide, residuals);
		if (isOpen && (TPolicy::INVALID_LABEL == atomicCAS(&(aGraph.label(secondVertex)), TPolicy::INVALID_LABEL, aCurrentLevel))) {
			aVertices.append(secondVertex);
		}

		/*int newQueueOffset = 0;
		int totalItems = 0;
		BlockScan(temp_storage).ExclusiveSum(shouldAdd, newQueueOffset, totalItems);
		if (threadIdx.x == 0) {
			currentQueueRunStart = mVertices.allocate(totalItems);
		}*/
	}
}

template<typename TGraph, typename TPolicy>
CUGIP_DECL_DEVICE void
propagateFromVertex2(int vertex, const int *aStartEnd, ParallelQueueView<int> &aVertices, TGraph &aGraph, int aCurrentLevel, const TPolicy &aPolicy)
{
	//typedef cub::BlockScan<int, TPolicy::THREADS> BlockScan;
	//__shared__ typename BlockScan::TempStorage temp_storage;

	int first = aStartEnd[2*threadIdx.x];//aGraph.neighborCount(vertex);
	int last = aStartEnd[2*threadIdx.x + 1];
	for (int i = first; i < last; ++i) {
		int secondVertex = aGraph.secondVertex(i);
		int label = aGraph.label(secondVertex);

		int connectionId = aGraph.connectionIndex(i);
		bool connectionSide = aGraph.connectionSide(i);
		auto residuals = aGraph.residuals(connectionId);
		bool isOpen = label == TPolicy::INVALID_LABEL && aPolicy.edgeTraversalCheck.invoke(connectionSide, residuals);
		if (isOpen && (TPolicy::INVALID_LABEL == atomicCAS(&(aGraph.label(secondVertex)), TPolicy::INVALID_LABEL, aCurrentLevel))) {
			aVertices.append(secondVertex);
			//printf("vertex count %d\n", aVertices.device_size());
		}
	}
}

struct NaiveWorkDistribution
{
	int start;
	int count;
	int gridSize;

	CUGIP_DECL_HYBRID
	NaiveWorkDistribution(int aStart, int aCount, int aGridSize)
		: start(aStart)
		, count(aCount)
		, gridSize(aGridSize)
	{}
};

template<typename TGraph, typename TPolicy>
struct NaiveSweepPass
{
	CUGIP_DECL_DEVICE void
	invoke(
		TGraph &aGraph,
		ParallelQueueView<int> &aVertices,
		/*int aStart,
		int aCount,*/
		NaiveWorkDistribution aWorkDistribution,
		int aCurrentLevel) const
	{
		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		int index = blockId * blockDim.x + threadIdx.x;

			//__syncthreads();
		while (index < aWorkDistribution.count) {
		//while (blockId * blockDim.x/*index*/ < aWorkDistribution.count) {
			if (index < aWorkDistribution.count) {
				int vertex = aVertices.get_device(aWorkDistribution.start + index);
				propagateFromVertex<TGraph, TPolicy>(vertex, aVertices, aGraph, aCurrentLevel, mPolicy);
			}
			//__syncthreads();
			index += blockDim.x * gridDim.x;
		}

	}

	TPolicy mPolicy;
};

template<typename TGraph, typename TPolicy>
struct NaiveSweepPass2
{
	CUGIP_DECL_DEVICE void
	invoke(
		TGraph &aGraph,
		ParallelQueueView<int> &aVertices,
		/*int aStart,
		int aCount,*/
		NaiveWorkDistribution aWorkDistribution,
		int aCurrentLevel)
	{
		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		int index = blockId * blockDim.x + threadIdx.x;

		__shared__ int listStartEnd[TPolicy::THREADS * 2];

		__syncthreads();
		while (index - threadIdx.x < aWorkDistribution.count) {
			int vertex = index < aWorkDistribution.count ? aVertices.get_device(aWorkDistribution.start + index) : 0;
			listStartEnd[threadIdx.x * 2] = vertex;
			listStartEnd[threadIdx.x * 2 + 1] = vertex + 1;
			//printf("AAAA\n");
			__syncthreads();
			listStartEnd[threadIdx.x] = aGraph.firstNeighborIndex(listStartEnd[threadIdx.x]);
			listStartEnd[threadIdx.x + blockDim.x] = aGraph.firstNeighborIndex(listStartEnd[threadIdx.x + blockDim.x]);
			//printf("BBBB\n");
			__syncthreads();
			if (index < aWorkDistribution.count) {
				propagateFromVertex2<TGraph, TPolicy>(vertex, listStartEnd, aVertices, aGraph, aCurrentLevel, mPolicy);
			}
			//printf("CCCC\n");
			index += blockDim.x * gridDim.x;
		}

		/*__syncthreads();
		if (threadIdx.x == 0)
			printf("vertex count %d\n", aVertices.device_size());*/
	}

	TPolicy mPolicy;
};

/*
template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
bfsNaivePropagationKernel(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		TGraph aGraph,
		int aCurrentLevel)
{
	NaiveSweepPass<TGraph, TPolicy>::invoke(
		aGraph,
		aVertices,
		aStart,
		aCount,
		aCurrentLevel);
}

template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
bfsNaivePropagationKernel_multi(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		ParallelQueueView<int> aLevelStarts,
		TGraph aGraph,
		int aCurrentLevel)
{

	int m = 0;
	do {
		__syncthreads();

		NaiveSweepPass<TGraph, TPolicy>::invoke(
			aGraph,
			aVertices,
			aStart,
			aCount,
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
	} while (m < TPolicy::MULTI_LEVEL_COUNT_LIMIT && aCount <= TPolicy::MULTI_LEVEL_LIMIT && aCount > 0);//(false);
}
*/

template<typename TGraph>
struct Tile {
	int vertexId;
	int listStart;
	int listLength;

	int listProgress;
	int progress;

	int totalCount;
	int prefixSum;

	CUGIP_DECL_DEVICE void
	load(const ParallelQueueView<int> &aVertices, int aOffset, int aCount)
	{
		if (threadIdx.x < aCount) {
			vertexId = aVertices.get_device(aOffset + threadIdx.x);
			//printf("vid %d, %d, %d\n", vertexId, aOffset, aOffset + threadIdx.x);
		} else {
			vertexId = -1;
		}
		progress = 0;
		listProgress = 0;
	}

	CUGIP_DECL_DEVICE void
	getAdjacencyList(TGraph &aGraph)
	{
		if (vertexId >= 0) {
			listStart = aGraph.firstNeighborIndex(vertexId);
			listLength = aGraph.firstNeighborIndex(vertexId + 1) - listStart;

			//printf("vid %d: %d %d\n", vertexId, listStart, listLength);
		} else {
			listStart = 0;
			listLength = 0;
		}
	}
};

template<typename TGraph>
CUGIP_DECL_DEVICE void
printTile(const Tile<TGraph> &aTile)
{
	printf("TID: %d; vertexId: %d; listStart: %d; listLength: %d; listProgress %d; progress %d\n",
		int(threadIdx.x),
		aTile.vertexId,
		aTile.listStart,
		aTile.listLength,
		aTile.listProgress,
		aTile.progress);
}


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


inline CUGIP_DECL_DEVICE void
printWorkLimits(const WorkLimits &aLimits)
{
	printf("TID: %d; elements: %d; offset: %d; guardedOffset: %d; guardedElements %d; outOfBounds %d\n",
		int(threadIdx.x),
		aLimits.elements,
		aLimits.offset,
		aLimits.guardedOffset,
		aLimits.guardedElements,
		aLimits.outOfBounds);
}

template<int tScheduleGranularity>
struct WorkDistribution
{
	int start;
	int count;
	int gridSize;

	int totalGrains;
	int grainsPerBlock;
	int extraGrains;

	CUGIP_DECL_HYBRID
	WorkDistribution(int aStart, int aCount, int aGridSize)
		: start(aStart)
		, count(aCount)
		, gridSize(aGridSize)
	{
		totalGrains = (count + tScheduleGranularity -1) / tScheduleGranularity;
		grainsPerBlock = totalGrains / gridSize;
		extraGrains = totalGrains - (grainsPerBlock * gridSize);
		/*CUGIP_DFORMAT(
			"WorkDistribution: \n\tstart: %1%"
			"\n\tcount:%2%"
			"\n\tgridSize: %3%"
			"\n\ttotalGrains: %4%"
			"\n\tgrainsPerBlock: %5%"
			"\n\textraGrains: %6%",
			start,
			count,
			gridSize,
			totalGrains,
			grainsPerBlock,
			extraGrains);*/
	}

	template<int tTileSize>
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
			workLimits.elements = count - (workLimits.offset - start);
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
	ParallelQueueView<int> &mVertices;
	// shared memory temporary buffer
	int *offsetScratch;
	//int *incomming;
	int mCurrentLevel;

	CUGIP_DECL_DEVICE
	TileProcessor(
		TGraph &aGraph,
		ParallelQueueView<int> &aVertices,
		int *aOffsetScratch,
		int currentLevel,
		const TPolicy &aPolicy
	)
		: mGraph(aGraph)
		, mVertices(aVertices)
		, offsetScratch(aOffsetScratch)
		, mCurrentLevel(currentLevel)
		, mPolicy(aPolicy)
	{
		//assert(false && "offsetScratch not initialized");
	}

	/*void
	processTile(int offset);*/

	CUGIP_DECL_DEVICE void
	processTile(int aOffset, int aCount)
	{
		typedef cub::BlockScan<int, TPolicy::THREADS> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;
		__shared__ int currentQueueRunStart;

		Tile<TGraph> tile = { 0 };
		tile.load(mVertices, aOffset, aCount);

		tile.getAdjacencyList(mGraph);
		//assert(mGraph.label(tile.vertexId) == mCurrentLevel-1);
		/*if (tile.vertexId >= 0 && mGraph.label(tile.vertexId) != mCurrentLevel -1) {
			printf("********** processing wrong id %d - %d (%d)\n", tile.vertexId, mGraph.label(tile.vertexId), mCurrentLevel - 1);
		}*/
		__syncthreads();
		/*printTile(tile);
		__syncthreads();*/

		//int total = 0;
		//int current = 0;
		tile.totalCount = 0;
		tile.prefixSum = 0;
		//BlockScan(temp_storage).ExclusiveSum(tile.listLength, current, total);
		BlockScan(temp_storage).ExclusiveSum(tile.listLength, tile.prefixSum, tile.totalCount);

		/*__syncthreads();
		printf("TID: %d, len: %d total: %d current: %d\n", int(threadIdx.x), tile.listLength, total, current);
		__syncthreads();*/

		tile.progress = 0;
		while (tile.progress < tile.totalCount) {
			expand(tile);
			__syncthreads();

			int scratchRemainder = min<int>(TPolicy::SCRATCH_ELEMENTS, tile.totalCount - tile.progress);
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
				//TODO - nema to delat jen jedno vlakno?
				//TODO - save to shared memory and then to global
				if (threadIdx.x == 0) {
					currentQueueRunStart = mVertices.allocate(totalItems);
				}
				__syncthreads();
				if (neighborId != -1) {

					mVertices.get_device(currentQueueRunStart + newQueueOffset) = neighborId;
				}
				/*printf("tid: %d, scratchOffset %d, scratchRemainder %d, neighborId %d total %d progress %d\n",
					threadIdx.x, scratchOffset, scratchRemainder, neighborId,
					tile.totalCount, tile.progress);*/
				__syncthreads();
			}
				/*if (threadIdx.x == 0) {
					printf("----------------------------------\n");
				}*/

			tile.progress += TPolicy::THREADS;
			__syncthreads();
		}
	}

	CUGIP_DECL_DEVICE int
	cullNeighbors(int scratchIndex) {
		//assert(scratchIndex >= 0);
		//assert(scratchIndex < TPolicy::SCRATCH_ELEMENTS);
		//assert(offsetScratch[scratchIndex] >= 0);
		//int source = incomming[scratchIndex];
		int neighborId = mGraph.secondVertex(offsetScratch[scratchIndex]);
		//printf("-- %d %d %d\n", neighborId, scratchIndex, offsetScratch[scratchIndex]);
		int label = mGraph.label(neighborId);
		int connectionId = mGraph.connectionIndex(offsetScratch[scratchIndex]);
		//int connectionId = mGraph.connectionIndex(neighborId);
		//if (connectionId < 0 || connectionId >= mGraph.mEdgeCount) {printf("before residuals %d %d - %d\n", connectionId, neighborId, label); }
		bool connectionSide = mGraph.connectionSide(offsetScratch[scratchIndex]);
		auto residuals = mGraph.residuals(connectionId);
		bool isClosed = label != TPolicy::INVALID_LABEL || !mPolicy.edgeTraversalCheck.invoke(connectionSide, residuals);
		if (isClosed || (TPolicy::INVALID_LABEL != atomicCAS(&(mGraph.label(neighborId)), TPolicy::INVALID_LABEL, mCurrentLevel))) {
			return -1;
		}
		return neighborId;
	}

	CUGIP_DECL_DEVICE void
	expand(Tile<TGraph> &aTile)
	{
		int scratchOffset = aTile.prefixSum + aTile.listProgress - aTile.progress; // ??
		while (aTile.listProgress < aTile.listLength && scratchOffset < TPolicy::SCRATCH_ELEMENTS) {
			offsetScratch[scratchOffset] = aTile.listStart + aTile.listProgress;
			//incomming[scratchOffset] = aTile.vertexId;
			/*printf("TID: %d; scratchOffset: %d; %d + %d\n",
				int(threadIdx.x),
				scratchOffset,
				aTile.listStart,
				aTile.listProgress);*/
			++aTile.listProgress;
			++scratchOffset;
		}
	}

	const TPolicy &mPolicy;
};

template<typename TGraph, typename TPolicy>
struct SweepPass
{
	CUGIP_DECL_DEVICE void
	invoke(
		TGraph &aGraph,
		ParallelQueueView<int> &aVertices,
		const WorkDistribution<TPolicy::SCHEDULE_GRANULARITY> &aWorkDistribution,
		int aCurrentLevel) const
	{
		__shared__ typename TPolicy::SharedMemoryData sharedMemoryData;

		WorkLimits workLimits = aWorkDistribution.template getWorkLimits<TPolicy::TILE_SIZE>();
		/*__syncthreads();
		printWorkLimits(workLimits);
		__syncthreads();*/
		if (workLimits.elements == 0) {
			return;
		}

		TileProcessor<TGraph, TPolicy> tileProcessor(aGraph, aVertices, sharedMemoryData.offsetScratch, aCurrentLevel, mPolicy);
		//tileProcessor.incomming = sharedMemoryData.incomming;
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

	TPolicy mPolicy;
};


/*template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
bfsPropagationKernel_b40c(
		ParallelQueueView<int> aVertices,
		WorkDistribution<TPolicy::SCHEDULE_GRANULARITY> aWorkDistribution,
		TGraph aGraph,
		int aCurrentLevel)
{
	SweepPass<TGraph, TPolicy>::invoke(
			aGraph,
			aVertices,
			aWorkDistribution,
			aCurrentLevel);
}*/

template<typename TGraph, typename TSweepOperator, typename TWorkDistribution>
CUGIP_GLOBAL void
bfsPropagationSingleIterationKernel(
		ParallelQueueView<int> aVertices,
		TWorkDistribution aWorkDistribution,
		TGraph aGraph,
		int aCurrentLevel,
		TSweepOperator aSweepOperator)
{
	aSweepOperator.invoke(
			aGraph,
			aVertices,
			aWorkDistribution,
			aCurrentLevel);
}

template<typename TGraph, typename TSweepOperator, typename TWorkDistribution>
CUGIP_GLOBAL void
bfsPropagationSingleIterationKernel2(
		ParallelQueueView<int> aVertices,
		TWorkDistribution aWorkDistribution,
		TGraph aGraph,
		int aCurrentLevel,
		TSweepOperator aSweepOperator
	)
{
	aSweepOperator.invoke(
			aGraph,
			aVertices,
			aWorkDistribution,
			aCurrentLevel);
}

template<typename TGraph, typename TSweepOperator, typename TWorkDistribution, typename TPolicy>
CUGIP_GLOBAL void
bfsPropagationMultiIterationKernel(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		ParallelQueueView<int> aLevelStarts,
		TGraph aGraph,
		int aCurrentLevel,
		TSweepOperator aSweepOperator)
{

	int m = 0;
	do {
		__syncthreads();
		aSweepOperator.invoke(
				aGraph,
				aVertices,
				TWorkDistribution(aStart, aCount, gridDim.x),
				aCurrentLevel);
		__syncthreads();
		int size = aVertices.device_size();
		aStart += aCount;
		aCount = size - aStart;
		if (threadIdx.x == 0) {
			aLevelStarts.append(size);
			//printf("size %d - %d\n", size, aCurrentLevel);
		}
		++aCurrentLevel;
		++m;
	} while (m < TPolicy::MULTI_LEVEL_COUNT_LIMIT && aCount <= TPolicy::MULTI_LEVEL_LIMIT && aCount > 0);//(false);
}

template<typename TGraph, typename TSweepOperator, typename TWorkDistribution, typename TPolicy>
CUGIP_GLOBAL void
bfsPropagationMultiIterationGlobalSyncKernel(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		ParallelQueueView<int> aLevelStarts,
		TGraph aGraph,
		int aCurrentLevel,
		TSweepOperator aSweepOperator,
		cub::GridBarrier barrier)
{

	int m = 0;
	do {
		__syncthreads();
		barrier.Sync();
		aSweepOperator.invoke(
				aGraph,
				aVertices,
				TWorkDistribution(aStart, aCount, gridDim.x),
				aCurrentLevel);
		__syncthreads();
		barrier.Sync();
		int size = aVertices.device_size();
		aStart += aCount;
		aCount = size - aStart;
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			aLevelStarts.append(size);
			//printf("size %d - %d\n", size, aCurrentLevel);
		}
		++aCurrentLevel;
		++m;
	} while (m < TPolicy::MULTI_LEVEL_COUNT_LIMIT && aCount <= TPolicy::MULTI_LEVEL_GLOBAL_SYNC_LIMIT && aCount > 0);
}

/*template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
bfsPropagationKernel_b40c_multi(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		ParallelQueueView<int> aLevelStarts,
		TGraph aGraph,
		int aCurrentLevel)
{

	int m = 0;
	do {
		__syncthreads();
		SweepPass<TGraph, TPolicy>::invoke(
				aGraph,
				aVertices,
				WorkDistribution<TPolicy::SCHEDULE_GRANULARITY>(aStart, aCount, gridDim.x),
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
	} while (m < TPolicy::MULTI_LEVEL_COUNT_LIMIT && aCount <= TPolicy::MULTI_LEVEL_LIMIT && aCount > 0);//(false);
}*/

/*
#if __CUDA_ARCH__ >= 350
template<typename TGraph, typename TPolicy>
CUGIP_GLOBAL void
bfsPropagationKernel_b40c_dynamic(
		ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		ParallelQueueView<int> aLevelStarts,
		TGraph aGraph,
		int aCurrentLevel,
		TPolicy aPolicy)
{

	int m = 0;
	do {
		if (threadIdx.x == 0) {
			dim3 blockSize1D(TPolicy::THREADS, 1, 1);
			dim3 levelGridSize1D(1 + (aCount - 1) / (blockSize1D.x), 1, 1);
			bfsPropagationKernel_b40c<TGraphData, TPolicy><<<levelGridSize1D, blockSize1D>>>(
				aVertices,
				WorkDistribution<TPolicy::SCHEDULE_GRANULARITY>(aStart, aCount, gridDim.x),
				aGraph,
				aCurrentLevel);
			cudaDeviceSynchronize();

			int size = aVertices.device_size();
			aStart += aCount;
			aCount = size - aStart;
			aLevelStarts.append(size);
		}
		__syncthreads();
		++aCurrentLevel;
		++m;
	} while (m < TPolicy::MULTI_LEVEL_COUNT_LIMIT && aCount > 0);
}*/

template<typename TGraphData, typename TPolicy>
CUGIP_GLOBAL void
relabel_compute_dynamic(
		TGraphData aGraph,
		ParallelQueueView<int> aVertexQueue,
		ParallelQueueView<int> aLevelStarts,
		TPolicy aPolicy)

		/*ParallelQueueView<int> aVertices,
		int aStart,
		int aCount,
		ParallelQueueView<int> aLevelStarts,
		TGraph aGraph,
		int aCurrentLevel,
		TPolicy aPolicy)*/
{
	#if __CUDA_ARCH__ >= 350

	if (threadIdx.x == 0) {
		typedef NaiveWorkDistribution JobDistribution;
		NaiveSweepPass2<TGraphData, TPolicy> sweepPass{aPolicy};

		int frontierSize = aVertexQueue.size();
		aLevelStarts.clear_device();
		aLevelStarts.push_back(0);
		aLevelStarts.push_back(frontierSize);
		//printf("LevelStarts %d, aVertexQueue %d\n", aLevelStarts.size(), aVertexQueue.size());
		bool finished = false;
		int currentLevel = 1;
		while (/*!finished && */currentLevel < aPolicy.maxLevels()) {
			//int frontierSize = aLevelStarts[currentLevel] - aLevelStarts[currentLevel - 1];

			if (frontierSize <= TPolicy::MULTI_LEVEL_COUNT_LIMIT) {
				dim3 blockSize1D(TPolicy::THREADS, 1, 1);
				dim3 levelGridSize1D(1, 1, 1);
				bfsPropagationMultiIterationKernel<TGraphData, NaiveSweepPass2<TGraphData, TPolicy>, JobDistribution, TPolicy><<<levelGridSize1D, blockSize1D>>>(
						aVertexQueue,
						aLevelStarts[currentLevel - 1],
						aLevelStarts[currentLevel] - aLevelStarts[currentLevel - 1],
						aLevelStarts,
						aGraph,
						currentLevel + 1,
						sweepPass);
				cudaDeviceSynchronize();
				currentLevel = aLevelStarts.size() - 1;
				frontierSize = aLevelStarts[currentLevel] - aLevelStarts[currentLevel - 1];
				if (frontierSize == 0) {
					break;
				}
			} else {
				dim3 blockSize1D(TPolicy::THREADS, 1, 1);
				dim3 levelGridSize1D(1 + (frontierSize - 1) / (blockSize1D.x), 1, 1);

				//printf("Level %d: %d -> %d\n", currentLevel, aLevelStarts[currentLevel - 1], aLevelStarts[currentLevel]);
				bfsPropagationSingleIterationKernel<TGraphData, NaiveSweepPass2<TGraphData, TPolicy>, JobDistribution><<<levelGridSize1D, blockSize1D>>>(
					aVertexQueue,
					JobDistribution(aLevelStarts[currentLevel - 1], frontierSize, levelGridSize1D.x),
					aGraph,
					currentLevel + 1,
					sweepPass
					);
				++currentLevel;
				cudaDeviceSynchronize();
				int lastLevelSize = aVertexQueue.size();
				frontierSize = lastLevelSize - aLevelStarts.back();
				//printf("size single %d - %d\n", lastLevelSize, currentLevel);
				//printf("lastLevelSize %d\n", lastLevelSize);
				if (frontierSize == 0) {
					break;
				}
				aLevelStarts.push_back(lastLevelSize);
			}
			/*bfsPropagationSingleIterationKernel<TGraphData, NaiveSweepPass<TGraphData, TPolicy>, JobDistribution><<<levelGridSize1D, blockSize1D>>>(
				aVertexQueue,
				JobDistribution(aLevelStarts[currentLevel - 1], frontierSize, levelGridSize1D.x),
				aGraph,
				currentLevel + 1,
				sweepPass);*/
			//CUGIP_CHECK_RESULT(cudaThreadSynchronize());


			//finished = ComputationStep<RelabelImplementation::Naive, true>::compute(*this, aGraph, currentLevel, aLevelStarts, aVertexQueue);
			//finished = ComputationStep<TPolicy::cRelabelImplementation, true>::compute(*this, aGraph, currentLevel, aLevelStarts, aVertexQueue, aPolicy);
			//finished = computation_step(aGraph, currentLevel, aLevelStarts, aVertexQueue);
			//CUGIP_DPRINT("Level = " << currentLevel << "; " << aVertexQueue.size());
		}
		//TODO - return value
		/*#else
			static_assert(false, "Compute capability 3.5 needed.");
		#endif  // __CUDA_ARCH__ >= 350*/
		/*int m = 0;
		do {
			if (threadIdx.x == 0) {
				dim3 blockSize1D(TPolicy::THREADS, 1, 1);
				dim3 levelGridSize1D(1 + (aCount - 1) / (blockSize1D.x), 1, 1);
				bfsPropagationKernel_b40c<TGraphData, TPolicy><<<levelGridSize1D, blockSize1D>>>(
					aVertices,
					WorkDistribution<TPolicy::SCHEDULE_GRANULARITY>(aStart, aCount, gridDim.x),
					aGraph,
					aCurrentLevel);
				cudaDeviceSynchronize();

				int size = aVertices.device_size();
				aStart += aCount;
				aCount = size - aStart;
				aLevelStarts.append(size);
			}
			__syncthreads();
			++aCurrentLevel;
			++m;
		} while (m < TPolicy::MULTI_LEVEL_COUNT_LIMIT && aCount > 0);*/
	}
	__syncthreads();
	#else
		assert(false && "__CUDA_ARCH__ not specified");
	#endif  // __CUDA_ARCH__ >= 350
}


template<typename TGraphData, typename TPolicy>
struct Relabel
{
	template<RelabelImplementation tImplementationId, bool tHelper>
	struct ComputationStep
	{
		static bool
		compute(Relabel<TGraphData, TPolicy> &aRelabel, TGraphData &aGraph, int &aCurrentLevel, thrust::host_vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue, const TPolicy &aPolicy)
		{
			int frontierSize = aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1];
			SweepPass<TGraphData, TPolicy> sweepPass{aPolicy};
			if (frontierSize <= TPolicy::MULTI_LEVEL_LIMIT) {
				//CUGIP_DFORMAT("MI frontier size: %1%", frontierSize);
				return aRelabel.template bfs_multi_iteration<SweepPass<TGraphData, TPolicy>, WorkDistribution<TPolicy::SCHEDULE_GRANULARITY>>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
			} else {
				//CUGIP_DFORMAT("SI frontier size: %1%", frontierSize);
				return aRelabel.template bfs_iteration<SweepPass<TGraphData, TPolicy>, WorkDistribution<TPolicy::SCHEDULE_GRANULARITY>>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
				//return aRelabel.bfs_iteration(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue);
			}
		}
	};

	template<bool tHelper>
	struct ComputationStep<RelabelImplementation::Naive, tHelper>
	{
		static bool
		compute(Relabel<TGraphData, TPolicy> &aRelabel, TGraphData &aGraph, int &aCurrentLevel, thrust::host_vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue, const TPolicy &aPolicy)
		{
			int frontierSize = aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1];
			NaiveSweepPass<TGraphData, TPolicy> sweepPass{aPolicy};
			if (frontierSize <= TPolicy::MULTI_LEVEL_LIMIT) {
				return aRelabel.template bfs_multi_iteration<NaiveSweepPass<TGraphData, TPolicy>, NaiveWorkDistribution>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
			} else {
				if (frontierSize <= TPolicy::MULTI_LEVEL_GLOBAL_SYNC_LIMIT) {
					//CUGIP_DFORMAT("MI frontier size: %1%", frontierSize);
					//return aRelabel.template bfs_multi_iteration<NaiveSweepPass<TGraphData, TPolicy>, NaiveWorkDistribution>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
					return aRelabel.template bfs_multi_iteration_global_sync<NaiveSweepPass<TGraphData, TPolicy>, NaiveWorkDistribution>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
				} else {
					//CUGIP_DFORMAT("SI frontier size: %1%", frontierSize);
					return aRelabel.template bfs_iteration<NaiveSweepPass<TGraphData, TPolicy>, NaiveWorkDistribution>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
					//return aRelabel.bfs_naive_iteration(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue);
				}
			}
		}
	};

	template<bool tHelper>
	struct ComputationStep<RelabelImplementation::OptimizedNaive, tHelper>
	{
		static bool
		compute(Relabel<TGraphData, TPolicy> &aRelabel, TGraphData &aGraph, int &aCurrentLevel, thrust::host_vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue, const TPolicy &aPolicy)
		{
			int frontierSize = aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1];
			NaiveSweepPass2<TGraphData, TPolicy> sweepPass{aPolicy};
			if (frontierSize <= TPolicy::MULTI_LEVEL_LIMIT) {
				//CUGIP_DFORMAT("MI frontier size: %1%", frontierSize);
				return aRelabel.template bfs_multi_iteration<NaiveSweepPass2<TGraphData, TPolicy>, NaiveWorkDistribution>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
			} else {
				//CUGIP_DFORMAT("SI frontier size: %1%", frontierSize);
				return aRelabel.template bfs_iteration<NaiveSweepPass2<TGraphData, TPolicy>, NaiveWorkDistribution>(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue, sweepPass);
				//return aRelabel.bfs_naive_iteration(aGraph, aCurrentLevel, aLevelStarts, aVertexQueue);
			}
		}
	};
	void
	compute(
		TGraphData &aGraph,
		ParallelQueueView<int> &aVertexQueue,
		thrust::host_vector<int> &aLevelStarts,
		const TPolicy &aPolicy)
	{
		init_bfs(aGraph, aVertexQueue);

		int lastLevelSize = aVertexQueue.size();
		//CUGIP_DPRINT("Level 1 size: " << lastLevelSize);
		aLevelStarts.clear();
		aLevelStarts.push_back(0);
		aLevelStarts.push_back(lastLevelSize);
		int currentLevel = 1;
		bool finished = lastLevelSize == 0;

		//CUGIP_DPRINT("Level = " << currentLevel << "; " << aVertexQueue.size());
		while (!finished && currentLevel < aPolicy.maxLevels()) {
			//finished = ComputationStep<RelabelImplementation::Naive, true>::compute(*this, aGraph, currentLevel, aLevelStarts, aVertexQueue);
			finished = ComputationStep<TPolicy::cRelabelImplementation, true>::compute(*this, aGraph, currentLevel, aLevelStarts, aVertexQueue, aPolicy);
			//finished = computation_step(aGraph, currentLevel, aLevelStarts, aVertexQueue);
			//CUGIP_DPRINT("Level = " << currentLevel << "; " << aVertexQueue.size());
		}
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}


	void
	compute_dynamic(
		TGraphData &aGraph,
		ParallelQueueView<int> &aVertexQueue,
		thrust::host_vector<int> &aLevelStarts,
		const TPolicy &aPolicy)
	{
		init_bfs(aGraph, aVertexQueue);
		//ParallelQueue<int> levelStarts;
		mLevelStartsQueue.reserve(2000);
		mLevelStartsQueue.clear();

		dim3 blockSize1D(32, 1, 1);
		dim3 levelGridSize1D(1, 1, 1);
		relabel_compute_dynamic<TGraphData, TPolicy><<<1, 1>>>(
		//relabel_compute_dynamic<TGraphData, TPolicy><<<levelGridSize1D, blockSize1D>>>(
			aGraph,
			aVertexQueue,
			mLevelStartsQueue.view(),
			aPolicy);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		//CUGIP_DPRINT("relabel compute dynamic done");

		mLevelStartsQueue.fill_host(aLevelStarts);
		/*if (aLevelStarts[aLevelStarts.size() - 1] ==  aLevelStarts[aLevelStarts.size() - 2]) {
			aLevelStarts.resize(aLevelStarts.size() - 1);
			mLevelStartsQueue.resize(aLevelStarts.size() - 1);
		}*/

		/*init_bfs(aGraph, aVertexQueue);
		int lastLevelSize = aVertexQueue.size();
		//CUGIP_DPRINT("Level 1 size: " << lastLevelSize);
		aLevelStarts.clear();
		aLevelStarts.push_back(0);
		aLevelStarts.push_back(lastLevelSize);
		int currentLevel = 1;
		bool finished = lastLevelSize == 0;

		//cudaProfilerStart();
		while (!finished && currentLevel < aPolicy.maxLevels()) {

			//finished = ComputationStep<RelabelImplementation::Naive, true>::compute(*this, aGraph, currentLevel, aLevelStarts, aVertexQueue);
			finished = ComputationStep<TPolicy::cRelabelImplementation, true>::compute(*this, aGraph, currentLevel, aLevelStarts, aVertexQueue, aPolicy);
			//finished = computation_step(aGraph, currentLevel, aLevelStarts, aVertexQueue);
			//CUGIP_DPRINT("Level = " << currentLevel << "; " << aVertexQueue.size());
		}
		*/

		//cudaProfilerStop();
	}

	/*bool
	bfs_multi_iteration_dynamic(
		TGraphData &aGraph,
		int &aCurrentLevel,
		std::vector<int> &aLevelStarts,
		ParallelQueueView<int> &aVertexQueue,
		TPolicy &aPolicy)
	{
		mLevelStartsQueue.reserve(TPolicy::MULTI_LEVEL_COUNT_LIMIT);
		dim3 blockSize1D(32, 1, 1);
		dim3 levelGridSize1D(1, 1, 1);
		mLevelStartsQueue.clear();

		bfsPropagationKernel_b40c_dynamic<TGraphData, TPolicy><<<levelGridSize1D, blockSize1D>>>(
				aVertexQueue,
				aLevelStarts[aCurrentLevel - 1],
				aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1],
				mLevelStartsQueue.view(),
				aGraph,
				aCurrentLevel + 1,
				aPolicy);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernelDynamic)");
		thrust::host_vector<int> starts;
		mLevelStartsQueue.fill_host(starts);
		int originalStart = aLevelStarts.back();
		int lastStart = originalStart;
		for (int i = 0; i < starts.size(); ++i) {
			if (starts[i] == lastStart) {
				lastStart = -1;
				break;
			} else {
				lastStart = starts[i];
			}
			aLevelStarts.push_back(starts[i]);
		}
		aCurrentLevel = aLevelStarts.size() - 1;
		//CUGIP_DPRINT("Level bundle " << (level + 1) << "-" << (aCurrentLevel + 1) << " size: " << (originalStart - aLevelStarts.back()));
		return (lastStart == originalStart) || (lastStart == -1);
	}*/

	void
	init_bfs(
		TGraphData &aGraph,
		ParallelQueueView<int> &aVertexQueue)
	{
		dim3 blockSize1D(TPolicy::THREADS, 1, 1);
		dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x), 1);

		aVertexQueue.clear();
		initBFSKernel<TGraphData, TPolicy><<<gridSize1D, blockSize1D>>>(aVertexQueue, aGraph);

		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		CUGIP_CHECK_ERROR_STATE("After initBFSKernel()");
	}

	template<typename TSweepOperator, typename TWorkDistribution>
	bool
	bfs_iteration(TGraphData &aGraph, int &aCurrentLevel, thrust::host_vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue, TSweepOperator &aSweepOperator)
	{
		int frontierSize = aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1];

		dim3 blockSize1D(TPolicy::THREADS, 1, 1);
		dim3 levelGridSize1D(1 + (frontierSize - 1) / (blockSize1D.x), 1, 1);
		CUGIP_CHECK_ERROR_STATE("Before bfsPropagationKernel()");

		bfsPropagationSingleIterationKernel<TGraphData, TSweepOperator, TWorkDistribution><<<levelGridSize1D, blockSize1D>>>(
			aVertexQueue,
			TWorkDistribution(aLevelStarts[aCurrentLevel - 1], frontierSize, levelGridSize1D.x),
			aGraph,
			aCurrentLevel + 1,
			aSweepOperator);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		++aCurrentLevel;
		CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel()");
		int lastLevelSize = aVertexQueue.size();
		if (lastLevelSize == aLevelStarts.back()) {
			return true;
		}
		aLevelStarts.push_back(lastLevelSize);
		return false;
	}

	template<typename TSweepOperator, typename TWorkDistribution>
	bool
	bfs_multi_iteration_global_sync(TGraphData &aGraph, int &aCurrentLevel, thrust::host_vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue, TSweepOperator &aSweepOperator)
	{
		mLevelStartsQueue.reserve(TPolicy::MULTI_LEVEL_COUNT_LIMIT);
		dim3 blockSize1D(TPolicy::THREADS, 1, 1);
		dim3 levelGridSize1D(TPolicy::MULTI_LEVEL_GLOBAL_SYNC_BLOCK_COUNT, 1, 1);
		mLevelStartsQueue.clear();

		cub::GridBarrierLifetime barrier;
		barrier.Setup(levelGridSize1D.x);
		bfsPropagationMultiIterationGlobalSyncKernel<TGraphData, TSweepOperator, TWorkDistribution, TPolicy><<<levelGridSize1D, blockSize1D>>>(
				aVertexQueue,
				aLevelStarts[aCurrentLevel - 1],
				aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1],
				mLevelStartsQueue.view(),
				aGraph,
				aCurrentLevel + 1,
				aSweepOperator,
				barrier);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel3)");
		thrust::host_vector<int> starts;
		mLevelStartsQueue.fill_host(starts);
		int originalStart = aLevelStarts.back();
		int lastStart = originalStart;
		for (int i = 0; i < starts.size(); ++i) {
			if (starts[i] == lastStart) {
				lastStart = -1;
				break;
			} else {
				lastStart = starts[i];
			}
			aLevelStarts.push_back(starts[i]);
		}
		aCurrentLevel = aLevelStarts.size() - 1;
		return (lastStart == originalStart) || (lastStart == -1);
	}


	template<typename TSweepOperator, typename TWorkDistribution>
	bool
	bfs_multi_iteration(TGraphData &aGraph, int &aCurrentLevel, thrust::host_vector<int> &aLevelStarts, ParallelQueueView<int> &aVertexQueue, TSweepOperator &aSweepOperator)
	{
		mLevelStartsQueue.reserve(TPolicy::MULTI_LEVEL_COUNT_LIMIT);
		dim3 blockSize1D(TPolicy::THREADS, 1, 1);
		dim3 levelGridSize1D(1, 1, 1);
		mLevelStartsQueue.clear();

		bfsPropagationMultiIterationKernel<TGraphData, TSweepOperator, TWorkDistribution, TPolicy><<<levelGridSize1D, blockSize1D>>>(
				aVertexQueue,
				aLevelStarts[aCurrentLevel - 1],
				aLevelStarts[aCurrentLevel] - aLevelStarts[aCurrentLevel - 1],
				mLevelStartsQueue.view(),
				aGraph,
				aCurrentLevel + 1,
				aSweepOperator);
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel3)");
		thrust::host_vector<int> starts;
		mLevelStartsQueue.fill_host(starts);
		int originalStart = aLevelStarts.back();
		int lastStart = originalStart;
		for (int i = 0; i < starts.size(); ++i) {
			if (starts[i] == lastStart) {
				lastStart = -1;
				break;
			} else {
				lastStart = starts[i];
			}
			aLevelStarts.push_back(starts[i]);
		}
		aCurrentLevel = aLevelStarts.size() - 1;
		return (lastStart == originalStart) || (lastStart == -1);
	}

	ParallelQueue<int> mLevelStartsQueue;
};

} // namespace cugip
