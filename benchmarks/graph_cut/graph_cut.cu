#include <cugip/advanced_operations/graph_cut.hpp>
#include <cugip/advanced_operations/detail/graph_cut_host_utils.hpp>

#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/detail/for_each_host.hpp>
#include <cugip/access_utils.hpp>

#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>

#include "graph_cut_trace_utils.hpp"

using namespace cugip;

CUGIP_GLOBAL void
debugFillSaturated(GraphCutData<float> aGraph, uint8_t *aBuffer)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		uint8_t value = 0;
		for (int i = aGraph.firstNeighborIndex(index); i < aGraph.firstNeighborIndex(index+1); ++i) {
			int connectionId = aGraph.connectionIndex(i);
			bool connectionSide = !aGraph.connectionSide(i);
			auto residuals = aGraph.residuals(connectionId);
			auto residual = residuals.getResidual(connectionSide);
			if (residual <= 0.0f) {
				value = 255;
			}
		}
		aBuffer[index] = value;
	}
}

template<typename THostArrayView>
void
debug_fill_saturated(const GraphCutData<float> &aGraph, THostArrayView aVertices)
{
	thrust::device_vector<uint8_t> tmp(aGraph.vertexCount());
	thrust::host_vector<uint8_t> tmp2(aGraph.vertexCount());

	dim3 blockSize1D( 512 );
	dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

	debugFillSaturated<<<gridSize1D, blockSize1D>>>(aGraph, thrust::raw_pointer_cast(tmp.data()));

	CUGIP_CHECK_ERROR_STATE("After pushThroughTLinksFromSourceKernel");
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	thrust::copy(tmp.begin(), tmp.end(), tmp2.begin());
	for (int i = 0; i < tmp2.size(); ++i) {
		aVertices[i] = tmp2[i];
	}
};

struct PrintNeighborhood
{
	CUGIP_DECL_DEVICE void
	operator()(int aIndex, GraphCutData<float> &aGraph)
	{
		int neighborCount = aGraph.neighborCount(aIndex);
		int firstNeighborIndex = aGraph.firstNeighborIndex(aIndex);
		for (int i = firstNeighborIndex; i < firstNeighborIndex + neighborCount; ++i) {
			int secondVertex = aGraph.secondVertex(i);
				//printf("%d - %d\n", vertex, secondVertex);

			int connectionId = aGraph.connectionIndex(i);
			bool connectionSide = aGraph.connectionSide(i);
			auto residuals = aGraph.residuals(connectionId);
			auto residual = residuals.getResidual(connectionSide);

			printf("%d; %d; %f\n", aIndex, secondVertex, residual);
		}
	}
};

struct TraceObject
{

	TraceObject(
		cugip::host_image_view<uint8_t, 3> aSaturated,
		cugip::host_image_view<uint8_t, 3> aExcess,
		cugip::host_image_view<float, 3> aLabels)
			: saturated(aSaturated)
			, excess(aExcess)
			, labels(aLabels)
	{}

	void
	computationStarted(cugip::GraphCutData<float> &aData)
	{
		/*thrust::host_vector<int> firstNeighbor(aData.vertexCount() + 1);
		thrust::device_ptr<int> ptr(aData.neighbors);
		thrust::copy(ptr, ptr + aData.vertexCount() + 1, firstNeighbor.begin());

		thrust::host_vector<int> secondVertices(2*aData.mEdgeCount);
		thrust::device_ptr<int> ptr2(aData.secondVertices);
		thrust::copy(ptr2, ptr2 + 2*aData.mEdgeCount, secondVertices.begin());

		thrust::host_vector<int> connectionIndices(2*aData.mEdgeCount);
		thrust::device_ptr<int> ptr3(aData.connectionIndices);
		thrust::copy(ptr3, ptr3 + 2*aData.mEdgeCount, connectionIndices.begin());

		thrust::host_vector<EdgeResidualsRecord<float>> residuals(aData.mEdgeCount);
		thrust::device_ptr<EdgeResidualsRecord<float>> ptr4(aData.mResiduals);
		thrust::copy(ptr4, ptr4 + aData.mEdgeCount, residuals.begin());

		for (int i = 0; i < aData.vertexCount(); ++i) {
			for (int j = firstNeighbor[i]; j < firstNeighbor[i+1]; ++j) {
				int connectionId = connectionIndices[j] & CONNECTION_INDEX_MASK;
				bool connectionSide = connectionIndices[j] & CONNECTION_VERTEX;
				auto res = residuals[connectionId];
				auto residual = res.getResidual(connectionSide);
				std::cout << i << "; " << secondVertices[j] << "; " << residual << "\n";
			}
		}*/
	}

	void
	beginIteration(int aIteration) {}

	void
	afterRelabel(int aIteration, const std::vector<int> &aLevelStarts) {}

	void
	afterPush(int aIteration, bool aDone, cugip::GraphCutData<float> &aGraph)
	{
		dim3 blockSize1D( 512 );
		dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

		pushThroughTLinksToSinkKernel<<<gridSize1D, blockSize1D>>>(aGraph);

		CUGIP_CHECK_ERROR_STATE("After push_through_tlinks_to_sink");
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());

		auto flow = thrust::reduce(
				thrust::device_pointer_cast(aGraph.mSinkFlow),
				thrust::device_pointer_cast(aGraph.mSinkFlow + aGraph.vertexCount()));

		BOOST_LOG_TRIVIAL(info) << "Iteration: " << aIteration << " Flow: " << flow;
	}

	void
	computationFinished(float aFlow, const cugip::GraphCutData<float> &aData)
	{
		debug_fill_saturated(aData, saturated);

		thrust::host_vector<float> tmp(aData.vertexCount());
		thrust::device_ptr<float> ptr(aData.vertexExcess);
		thrust::copy(ptr, ptr + aData.vertexCount(), tmp.begin());

		thrust::host_vector<int> tmpLabels(aData.vertexCount());
		thrust::device_ptr<int> ptr2(aData.labels);
		thrust::copy(ptr2, ptr2 + aData.vertexCount(), tmpLabels.begin());

		for (int i = 0; i < tmp.size(); ++i) {
			excess[i] = tmp[i] > 0.0f ? 255 : 0;
			labels[i] = tmpLabels[i];
			CUGIP_ASSERT(tmp[i] <= 0.0f || (labels[i] < 0 || labels[i] >= tmp.size()));
		}
	}
	cugip::host_image_view<uint8_t, 3> saturated;
	cugip::host_image_view<uint8_t, 3> excess;
	cugip::host_image_view<float, 3> labels;
};


void
computeCudaGraphCutImplementation(const cugip::GraphData<float> &aGraphData, cugip::host_image_view<uint8_t, 3> aOutput, uint8_t aMaskValue, CudacutConfig &aConfig)
{
	cugip::Graph<float> graph;
	graph.set_vertex_count(aGraphData.tlinksSource.size());

	graph.set_nweights(
		aGraphData.edges.size(),
		aGraphData.edges.data(),
		aGraphData.weights.data(),
		aGraphData.weightsBackward.data());

	graph.set_tweights(
		aGraphData.tlinksSource.data(),
		aGraphData.tlinksSink.data());

	BOOST_LOG_TRIVIAL(info) << "Computing max flow ...";
	boost::timer::cpu_timer computationTimer;
	computationTimer.start();

	TraceObject traceObject(
			aConfig.saturated,
			aConfig.excess,
			aConfig.labels);

	float flow = graph.max_flow();
	//float flow = graph.max_flow_with_tracing(traceObject);
	computationTimer.stop();
	BOOST_LOG_TRIVIAL(info) << "Max flow: " << flow;
	//BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");
	//BOOST_LOG_TRIVIAL(info) << "Computation time: " << boost::timer::format(computationTimer.elapsed(), 9, "%w");
	BOOST_LOG_TRIVIAL(info) << "Computation time: " << (computationTimer.elapsed().wall / 1000000000.0f);
	BOOST_LOG_TRIVIAL(info) << "Filling output ...";
	graph.fill_segments(linear_access_view(aOutput), aMaskValue, 0);


}
