#include <cugip/advanced_operations/detail/graph_cut_host_utils.hpp>

#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/detail/for_each_host.hpp>

#include <boost/log/trivial.hpp>

#include "graph_cut_trace_utils.hpp"


constexpr float cTLinkWeight = 1000000.f;
//constexpr float cTLinkWeight = 1.0e10f;

void
computeCudaGraphCutImplementation(const cugip::GraphData<float> &aGraphData, cugip::host_image_view<uint8_t, 3> aOutput, uint8_t aMaskValue, CudacutConfig &aConfig);

void
computeCudaGraphCut(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const uint8_t, 3> aMarkers,
	cugip::host_image_view<uint8_t, 3> aOutput,
	float aSigma,
	uint8_t aMaskValue,
	CudacutConfig &aConfig)
{
	using namespace cugip;

	VonNeumannNeighborhood<3> neighborhood;

	int edgeCount = product(aData.dimensions()) * neighborhood.size() / 2;
	int vertexCount = product(aData.dimensions());

	GraphData<float> graphData;
	graphData.reserve(vertexCount, edgeCount);
	graphData.setVertexCount(vertexCount);

	auto size = aData.dimensions();
	BOOST_LOG_TRIVIAL(info) << "Setting edge weights...";
	region<3> imageRegion{ Int3(), size };
	for_each(
		imageRegion,
		[&](const Int3 &coordinate) {
			int centerIdx = get_blocked_order_access_index<2>(size, coordinate);
			//int centerIdx = get_linear_access_index(size, coordinate);
			float source_weight = aMarkers[coordinate] == 128 ? cTLinkWeight : 0.0f;
			float sink_weight = aMarkers[coordinate] == 255 ? cTLinkWeight : 0.0f;
			if (min(coordinate) <= 1 || min(size - coordinate) <= 2) {
				source_weight = cTLinkWeight;
				sink_weight = 0.0f;
			}
			graphData.setTWeights(centerIdx, source_weight, sink_weight);
			for (int n = 1; n < (neighborhood.size() + 1) / 2; ++n) {
				Int3 neighbor = coordinate + neighborhood.offset(n);
				int neighborIdx = get_blocked_order_access_index<2>(size, neighbor);
				//int neighborIdx = get_linear_access_index(size, neighbor);
				if (isInsideRegion(aData.dimensions(), neighbor)) {
					float weight =  std::exp(-sqr(aData[coordinate] - aData[neighbor]) / 2.0f * sqr(aSigma));
					graphData.addEdge(neighborIdx, centerIdx, weight, weight);
				}
			}

		});

	/*cugip::Graph<float> graph;
	graph.set_vertex_count(vertexCount);

	graph.set_nweights(
		graphData.edges.size(),
		graphData.edges.data(),
		graphData.weights.data(),
		graphData.weightsBackward.data());

	graph.set_tweights(
		graphData.tlinksSource.data(),
		graphData.tlinksSink.data());

	float flow = graph.max_flow();*/
	BOOST_LOG_TRIVIAL(info) << "Computing max flow ...";
	computeCudaGraphCutImplementation(graphData, aOutput, aMaskValue, aConfig);
}