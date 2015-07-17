#define BOOST_TEST_MODULE GraphCutTest
#include <boost/test/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/advanced_operations/detail/graph_cut_relabeling.hpp>
#include <cugip/advanced_operations/graph_cut.hpp>
#include <cugip/parallel_queue.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

void
fillGridGraph(cugip::Graph<float> &aGraph, int aSize, int aSeparationSize)
{
	using namespace cugip;
	aGraph.set_vertex_count(aSize * aSize);

}

void
fillLayeredGraph(cugip::Graph<float> &aGraph, int aLayerCount, int aLayerSize, int aNeighborCount, bool sourceSinkFlip)
{
	using namespace cugip;
	aGraph.set_vertex_count(aLayerCount * aLayerSize);

	std::vector<EdgeRecord> edges((aLayerCount - 1) * aLayerSize * aNeighborCount);

	int edgeIdx = 0;
	std::vector<float> forwardWeights(edges.size());
	std::vector<float> backwardWeights(edges.size());
	for (int layer = 1; layer < aLayerCount; ++layer) {
		float weight = (layer == aLayerCount / 2) ? 1.0f : 20.0f;
		for (int i = 0; i < aLayerSize; ++i) {
			for (int j = 0; j < aNeighborCount; ++j) {
				int firstVertex = (layer - 1) * aLayerSize + i;
				int secondVertex = layer * aLayerSize + ((i + j) % aLayerSize);
				edges[edgeIdx] = EdgeRecord(firstVertex, secondVertex);
				forwardWeights[edgeIdx] = weight;
				backwardWeights[edgeIdx] = 2*weight;
				//std::cout << firstVertex << "; " << secondVertex << "\n";
				++edgeIdx;
			}
		}
	}

	std::vector<float> tweightsSource(aLayerCount * aLayerSize);
	std::vector<float> tweightsSink(aLayerCount * aLayerSize);
	std::fill(begin(tweightsSource), begin(tweightsSource) + aLayerSize, 100000.0f);
	std::fill(end(tweightsSink) - aLayerSize, end(tweightsSink), 100000.0f);

	aGraph.set_nweights(
			edges.size(),
			edges.data(),
			forwardWeights.data(),
			backwardWeights.data());
	if (sourceSinkFlip) {
		aGraph.set_tweights(
			tweightsSink.data(),
			tweightsSource.data());
	} else {
		aGraph.set_tweights(
			tweightsSource.data(),
			tweightsSink.data());
	}
}


void
fillTwoLayerGraph(cugip::Graph<float> &aGraph, int aVertexCount, int aLayerSize, int aNeighborCount)
{
	using namespace cugip;
	aGraph.set_vertex_count(aVertexCount);

	std::vector<EdgeRecord> edges(aLayerSize * aNeighborCount);

	CUGIP_DPRINT("Edge count " << edges.size());

	int edgeIdx = 0;
	for (int i = 0; i < aLayerSize; ++i) {
		for (int j = 0; j < aNeighborCount; ++j) {
			int secondVertex = aLayerSize + ((i + j) % aLayerSize);
			edges[edgeIdx++] = EdgeRecord(i, secondVertex);
			std::cout << i << "; " << secondVertex << "\n";
		}
	}
	std::vector<float> weights(edges.size());
	std::vector<float> tweights(aVertexCount);
	std::fill(begin(weights), end(weights), 1.0f);
	std::fill(begin(tweights), begin(tweights) + aLayerSize, 100000.0f);
	aGraph.set_nweights(
			edges.size(),
			edges.data(),
			weights.data(),
			weights.data());
	aGraph.set_tweights(
			tweights.data(),
			tweights.data());
}

BOOST_AUTO_TEST_CASE(bfsPropagation)
{
	using namespace cugip;
	typedef GraphCutPolicy::RelabelPolicy<16, 8> Policy;
	Graph<float> graph;
	static const int cVertexCount = 32;
	static const int cLayerSize = cVertexCount >> 1;
	static const int cNeighborCount = 4;

	fillTwoLayerGraph(graph, cVertexCount, cLayerSize, cNeighborCount);

	GraphCutData<float> graphCutData;
	graphCutDataFromGraph(graph, graphCutData);

	ParallelQueue<int> queue;
	queue.reserve(cVertexCount * cNeighborCount);
	queue.clear();
	CUGIP_DPRINT("AAAAAAAAAA");

	dim3 blockSize1D(Policy::THREADS, 1, 1);
	dim3 graphGridSize1D(1 + (cVertexCount - 1) / (blockSize1D.x), 1, 1);
	try {
	initResidualsKernel<<<graphGridSize1D, blockSize1D>>>(
			thrust::raw_pointer_cast(graph.mEdgeWeightsForward.data()),
			thrust::raw_pointer_cast(graph.mEdgeWeightsBackward.data()),
			thrust::raw_pointer_cast(graphCutData.mResiduals),
			graphCutData.mEdgeCount);
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	initBFSKernel<GraphCutData<float>, Policy><<<graphGridSize1D, blockSize1D>>>(
			queue.view(),
			graphCutData);
	CUGIP_CHECK_ERROR_STATE("After initBFSKernel");
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	BOOST_CHECK_EQUAL(queue.size(), cLayerSize);

	CUGIP_DPRINT("queue size " << queue.size());
	CUGIP_DPRINT("BBBBBBBBBB");
	int frontierSize = queue.size();
	dim3 levelGridSize1D(1 + (frontierSize - 1) / (blockSize1D.x), 1, 1);
	bfsPropagationKernel_b40c<GraphCutData<float>, Policy><<<levelGridSize1D, blockSize1D>>>(
			queue.view(),
			WorkDistribution(0, frontierSize, levelGridSize1D.x, Policy::SCHEDULE_GRANULARITY),
			graphCutData,
			1);
	CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel_b40c");
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	thrust::host_vector<int> labels = graph.mLabels;
	for (int i = 0; i < labels.size(); ++i) {
		std::cout << i << " - " << labels[i] << std::endl;
	}
	std::cout << "----------------------------------------\n";
	thrust::host_vector<EdgeResidualsRecord<float>> residuals = graph.mResiduals;
	for (int i = 0; i < residuals.size(); ++i) {
		std::cout << i << ") " << residuals[i].residuals[0] << " = " << residuals[i].residuals[1] << std::endl;
	}
	/*thrust::host_vector<int> tmp = queue.mBuffer;
	for (int i = cLayerSize; i < queue.size(); ++i) {
		std::cout << tmp[i] << "; ";
	}*/
	std::cout << "\n";
	} catch (...) {
		CUGIP_DPRINT("QQQQQQQQQQQQQ");
	}

	//CUGIP_DPRINT("queue size " << queue.size());
}

/*
BOOST_AUTO_TEST_CASE(MinCutLayeredGraph)
{
	using namespace cugip;

	Graph<float> graph;
	//fillLayeredGraph(graph, 100, 1024, 8);
	fillLayeredGraph(graph, 100, 1024, 8, false);

	float flow = graph.max_flow();
	CUGIP_DPRINT("Computed flow " << flow);
	BOOST_CHECK_EQUAL(flow, 1024 * 8);
}


BOOST_AUTO_TEST_CASE(MinCutLayeredGraph2)
{
	using namespace cugip;

	Graph<float> graph;
	//fillLayeredGraph(graph, 100, 1024, 8);
	fillLayeredGraph(graph, 100, 1024, 8, true);

	float flow = graph.max_flow();
	CUGIP_DPRINT("Computed flow " << flow);
	BOOST_CHECK_EQUAL(flow, 2*1024 * 8);
}
*/
