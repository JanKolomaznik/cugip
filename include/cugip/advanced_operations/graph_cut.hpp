#pragma once


#include <cugip/math.hpp>
#include <cugip/memory.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/device_flag.hpp>
#include <limits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>


#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>

#include <fstream>

#include <cugip/advanced_operations/detail/graph_cut_implementation.hpp>

namespace cugip {

typedef unsigned NodeId;
typedef unsigned long CombinedNodeId;
struct EdgeRecord
{
	__host__ __device__
	EdgeRecord( NodeId aFirst, NodeId aSecond )
	{
		first = min( aFirst, aSecond );
		second = max( aFirst, aSecond );
	}
	__host__ __device__
	EdgeRecord(): edgeCombIdx(0)
	{ }

	union {
		CombinedNodeId edgeCombIdx;
		struct {
			NodeId second;
			NodeId first;
		};
	};
};


template <typename TFlow>
class Graph
{
public:
	typedef int VertexId;
	typedef TFlow EdgeWeight;

	void
	set_vertex_count(size_t aCount);

	void
	set_nweights(
		size_t aEdgeCount,
		EdgeRecord *aEdges,
		EdgeWeight *aWeightsForward,
		EdgeWeight *aWeightsBackward);

	void
	set_tweights(
		EdgeWeight *aCapSource,
		EdgeWeight *aCapSink);

	TFlow
	max_flow() {
		MinimalGraphCutComputation<Graph<TFlow>> maxFlowComputation;

		maxFlowComputation.setGraph(*this);

		return maxFlowComputation.run();
	}

	void
	debug_print();
//protected:
	/*void
	init_residuals();

	bool
	bfs_iteration(size_t &aCurrentLevel);
	void
	assign_label_by_distance();

	void
	push_through_tlinks_from_source();

	void
	push_through_tlinks_to_sink();

	bool
	push();*/

	GraphCutData<TFlow> mGraphData;

	std::vector<int> mLevelStarts;
	ParallelQueue<int> mVertexQueue;
	ParallelQueue<int> mLevelStartsQueue;

	thrust::device_vector<EdgeWeight> mSourceTLinks; // n
	thrust::device_vector<EdgeWeight> mSinkTLinks; // n
	thrust::device_vector<EdgeWeight> mExcess; // n
	thrust::device_vector<int> mLabels; // n
	thrust::device_vector<int> mNeighbors; // n

	thrust::device_vector<int> mSecondVertices; // 2*m
	thrust::device_vector<int> mEdges; // 2*m
	thrust::device_vector<EdgeResidualsRecord<TFlow> > mResiduals; // m

	thrust::device_vector<EdgeWeight> mEdgeWeightsForward; // m
	thrust::device_vector<EdgeWeight> mEdgeWeightsBackward; // m

	thrust::device_vector<EdgeWeight> mSinkFlow; // n
};



/*
template<typename TFlow>
bool
Graph<TFlow>::push()
{
	//CUGIP_DPRINT("push()");
	dim3 blockSize1D(512);
	dim3 gridSize1D((mGraphData.vertexCount() + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);
	device_flag pushSuccessfulFlag;
	size_t pushIterations = 0;
	do {
		pushSuccessfulFlag.reset_host();
		++pushIterations;
		pushKernel<<<gridSize1D, blockSize1D>>>(mGraphData, pushSuccessfulFlag.view());
		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After pushKernel()");
	} while (pushSuccessfulFlag.check_host());
	CUGIP_DPRINT("Push iterations " << pushIterations);
	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After push()");
	//push_through_tlinks_to_sink(); //TODO implement

	//push_through_tlinks_to_source();

	return pushIterations > 1;
}*/

/*
template<typename TFlow>
TFlow
Graph<TFlow>::max_flow()
{
	boost::timer::cpu_timer timer;
	timer.start();
	//CUGIP_DPRINT("MAX FLOW");
	init_residuals();
	push_through_tlinks_from_source();

	//debug_print();
	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After max_flow init");
	timer.stop();
	std::cout << timer.format(9, "%w") << "\n";
	bool done = false;
	size_t iteration = 0;
	while(!done) {
		timer.start();
		assign_label_by_distance();
		//debug_print();
		done = !push();
		//if (iteration >10) break;
	//push_through_tlinks_to_sink();
		//CUGIP_DPRINT("**iteration " << iteration << "; flow = " << thrust::reduce(mSinkFlow.begin(), mSinkFlow.end()));
		timer.stop();
		CUGIP_DPRINT("**iteration " << iteration << ": " << timer.format(9, "%w"));
		++iteration;
	}
	push_through_tlinks_to_sink();
	//CUGIP_DPRINT("Used iterations: " << iteration);
	return thrust::reduce(mSinkFlow.begin(), mSinkFlow.end());
}*/


template<typename TFlow>
void
Graph<TFlow>::set_vertex_count(size_t aCount)
{
	mSourceTLinks.resize(aCount);
	mSinkTLinks.resize(aCount);
	mExcess.resize(aCount);
	mLabels.resize(aCount);
	mNeighbors.resize(aCount);
	mSinkFlow.resize(aCount);

	mGraphData.vertexExcess = thrust::raw_pointer_cast(&mExcess[0]); // n
	mGraphData.labels = thrust::raw_pointer_cast(&mLabels[0]);; // n
	mGraphData.mSourceTLinks = thrust::raw_pointer_cast(&mSourceTLinks[0]);// n
	mGraphData.mSinkTLinks = thrust::raw_pointer_cast(&mSinkTLinks[0]);// n
	mGraphData.mVertexCount = aCount;

	mVertexQueue.reserve(aCount);
	mVertexQueue.clear();
	CUGIP_DPRINT("Vertex queue size: " << aCount);

	mLevelStartsQueue.reserve(500);
}

template<typename TFlow>
void
Graph<TFlow>::set_nweights(
	size_t aEdgeCount,
	EdgeRecord *aEdges,
	EdgeWeight *aWeightsForward,
	EdgeWeight *aWeightsBackward)
{
	std::vector<std::vector<std::pair<int, int> > > edges(mLabels.size());
	for (size_t i = 0; i < aEdgeCount; ++i) {
		EdgeRecord &edge = aEdges[i];
		//std::cout << edge.first << "; " << edge.second << std::endl;
		edges.at(edge.first).push_back(std::make_pair(int(i), edge.second));
		edges.at(edge.second).push_back(std::make_pair(int(i), edge.first));
	}
	thrust::host_vector<int> neighbors(mLabels.size());
	thrust::host_vector<int> secondVertices(2 * aEdgeCount);
	thrust::host_vector<int> edgeIndex(2 * aEdgeCount);

	int start = 0;
	for (size_t i = 0; i < edges.size(); ++i) {
		neighbors[i] = start;
		for (size_t j = 0; j < edges[i].size(); ++j) {
			bool connectionSide = i < edges[i][j].second;

			secondVertices[start + j] = edges[i][j].second;
			edgeIndex[start + j] = connectionSide ? /*CONNECTION_VERTEX | */edges[i][j].first : edges[i][j].first;
		}
		start += edges[i].size();
	}
	neighbors.push_back(neighbors.back());
	//std::copy(begin(neighbors), end(neighbors), std::ostream_iterator<int>(std::cout, " "));
	//std::copy(begin(edgeIndex), end(edgeIndex), std::ostream_iterator<int>(std::cout, " "));

	mNeighbors = neighbors;
	mGraphData.neighbors = thrust::raw_pointer_cast(&mNeighbors[0]);
	mSecondVertices = secondVertices;
	mGraphData.secondVertices = thrust::raw_pointer_cast(&mSecondVertices[0]);
	mEdges = edgeIndex;
	mGraphData.connectionIndices = thrust::raw_pointer_cast(&mEdges[0]);

	mEdgeWeightsForward.resize(aEdgeCount);
	mEdgeWeightsBackward.resize(aEdgeCount);
	thrust::copy(aWeightsForward, aWeightsForward + aEdgeCount, mEdgeWeightsForward.begin());
	thrust::copy(aWeightsBackward, aWeightsBackward + aEdgeCount, mEdgeWeightsBackward.begin());

	mResiduals.resize(aEdgeCount);
	mGraphData.mResiduals = thrust::raw_pointer_cast(&mResiduals[0]);
	//throw 3;
}

template<typename TFlow>
void
Graph<TFlow>::set_tweights(
	EdgeWeight *aCapSource,
	EdgeWeight *aCapSink)
{
	thrust::copy(aCapSource, aCapSource + mSourceTLinks.size(), mSourceTLinks.begin());
	thrust::copy(aCapSink, aCapSink + mSinkTLinks.size(), mSinkTLinks.begin());

	thrust::fill(mSinkFlow.begin(), mSinkFlow.end(), 0.0f);
}


template<typename TFlow>
void
Graph<TFlow>::debug_print()
{
	thrust::host_vector<EdgeWeight> excess = mExcess;
	thrust::host_vector<int> labels = mLabels;

	for (size_t i = 0; i < excess.size(); ++i) {
		//if (excess[i] > 0)
			std::cout << i << ": " << excess[i] << " - " << labels[i] << "\n";
	}
}

} //namespace cugip


