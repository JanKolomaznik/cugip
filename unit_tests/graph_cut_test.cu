#define BOOST_TEST_MODULE UtilsTest
#include <boost/test/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/advanced_operations/detail/graph_cut_relabeling.hpp>
#include <cugip/advanced_operations/graph_cut.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

CUGIP_GLOBAL void
testBlockScanIn(int *output)
{
	__shared__ int buffer[512 + 1];

	auto sum = cugip::block_prefix_sum_in<int>(threadIdx.x, 512, threadIdx.x, buffer);
	if (threadIdx.x == 0) {
		printf("Total %d\n", sum.total);
	}
	__syncthreads();
	buffer[threadIdx.x] = - 1;
	output[threadIdx.x] = sum.current - (threadIdx.x + 1) * threadIdx.x / 2;
}



CUGIP_GLOBAL void
testBlockScanEx(int *output)
{
	__shared__ int buffer[512 + 1];

	auto sum = cugip::block_prefix_sum_ex<int>(threadIdx.x, 512, threadIdx.x, buffer);
	if (threadIdx.x == 0) {
		printf("Total %d\n", sum.total);
	}
	__syncthreads();
	buffer[threadIdx.x] = - 1;
	output[threadIdx.x] = sum.current - (threadIdx.x + 1) * threadIdx.x / 2 + threadIdx.x;
}

BOOST_AUTO_TEST_CASE(bfsPropagation)
{
	using namespace cugip;
	typedef GraphCutPolicy::RelabelPolicy Policy;
	Graph<float> graph;
	static const int cVertexCount = 1024;
	static const int cLayerSize = cVertexCount >> 1;
	static const int cNeighborCount = 16;
	graph.set_vertex_count(cVertexCount);

	std::vector<EdgeRecord> edges(cLayerSize * cNeighborCount);


	int edgeIdx = 0;
	for (int i = 0; i < cLayerSize; ++i) {
		for (int j = 0; j < cNeighborCount; ++j) {
			int secondVertex = cLayerSize + ((i + j) % cLayerSize);
			edges[edgeIdx] = EdgeRecord(i, secondVertex);
		}
	}
	std::vector<float> weights(edges.size());
	std::vector<float> tweights(cVertexCount);
	std::fill(begin(weights), end(weights), 1.0f);
	std::fill(begin(tweights), end(tweights), 1.0f);
	graph.set_nweights(
			edges.size(),
			edges.data(),
			weights.data(),
			weights.data());
	graph.set_tweights(
			tweights.data(),
			tweights.data());

	GraphCutData<flow> graphCutData;
	graphCutDataFromGraph(graph, graphCutData);

	dim3 blockSize1D(Policy::THREADS, 1, 1);
	int frontierSize = cLayerSize;
	dim3 levelGridSize1D(1 + (frontierSize - 1) / (blockSize1D.x), 1, 1);

	bfsPropagationKernel_b40c<TGraphData, Policy><<<levelGridSize1D, blockSize1D>>>(
			aVertexQueue,
			WorkDistribution(aLevelStarts[aCurrentLevel - 1], frontierSize, levelGridSize1D.x, TPolicy::SCHEDULE_GRANULARITY),
			aGraph,
			aCurrentLevel + 1);

	/*thrust::device_vector<int> buffer(512);

	testBlockScanIn<<<1, 512>>>(thrust::raw_pointer_cast(buffer.data()));
	cudaThreadSynchronize();

	int result = thrust::reduce(buffer.begin(), buffer.end(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(result, 0);*/
}

