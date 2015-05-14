#pragma once


namespace cugip {

template<typename TFlow>
void
assignLabelByDistance(GraphCutData<TFlow> &aGraphData, ParallelQueueView<int> aVertices)
{
	dim3 blockSize1D(512, 1, 1);
	dim3 gridSize1D((aGraphData.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x), 1);

	//mVertexQueue.clear();
	initBFSKernel<<<gridSize1D, blockSize1D>>>(aVertices, aGraphData);

	cudaThreadSynchronize();
	int lastLevelSize = mVertexQueue.size();
	mLevelStarts.clear();
	mLevelStarts.push_back(0);
	mLevelStarts.push_back(lastLevelSize);
	size_t currentLevel = 1;
	bool finished = lastLevelSize == 0;
	while (!finished) {
		finished = bfs_iteration(currentLevel);
	}
}


template<typename TFlow>
CUGIP_GLOBAL void
maxFlowIterationsKernel(GraphCutData<TFlow> aGraph, ParallelQueueView<int> aVertices)
{
	if (threadIdx.x == 0) {
		bool done = false;
		while(!done) {
			assignLabelByDistance(aGraph, aVertices);
			done = !push();
		}
	}
	__syncthreads();
}

} //namespace cugip
