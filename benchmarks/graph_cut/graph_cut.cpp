#include <cugip/advanced_operations/detail/graph_cut_host_utils.hpp>

#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/for_each.hpp>

#include <boost/log/trivial.hpp>

#include "graph_cut_trace_utils.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>

#include "graph.hpp"

namespace ba = boost::accumulators;

constexpr float cTLinkWeight = 1000000.f;
//constexpr float cTLinkWeight = 1.0e10f;

void
computeCudaGraphCutImplementation(const cugip::GraphData<float> &aGraphData, CudacutSimpleConfig &aConfig, std::vector<int> &aMaskedMarkers);

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
	ba::accumulator_set< double, ba::features<ba::tag::min, ba::tag::max, ba::tag::mean>> acc;
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
					acc(weight);
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
	BOOST_LOG_TRIVIAL(info) << boost::str(boost::format("Edge weights statistics: min %1%, max %2%, mean %3%") % ba::extract_result<ba::tag::min>(acc) % ba::extract_result<ba::tag::max>(acc) % ba::extract_result<ba::tag::mean>(acc));
	BOOST_LOG_TRIVIAL(info) << "Computing max flow ...";
	computeCudaGraphCutImplementation(graphData, aOutput, aMaskValue, aConfig);
}

std::vector<int>
computeCudaGraphCut(
	const GraphStats &aGraph,
	const std::vector<std::pair<bool, int>> &aMarkers,
	CudacutSimpleConfig aConfig)
{
	using namespace cugip;
	int edgeCount = aGraph.edges.size();//product(aData.dimensions()) * neighborhood.size() / 2;
	int vertexCount = aGraph.nodes.size();// product(aData.dimensions());

	GraphData<float> graphData;
	graphData.reserve(vertexCount, edgeCount);
	graphData.setVertexCount(vertexCount);
	BOOST_LOG_TRIVIAL(info) << "Edge count " << edgeCount;
	BOOST_LOG_TRIVIAL(info) << "Vertex count " << vertexCount;

	BOOST_LOG_TRIVIAL(info) << "Adding edges ...";
	ba::accumulator_set< double, ba::features<ba::tag::min, ba::tag::max, ba::tag::mean>> acc;
	for (const auto &edge : aGraph.edges) {
		acc(edge.second.sum);
		graphData.addEdge(
			edge.first[0]-1,
			edge.first[1]-1,
			edge.second.sum,
			edge.second.sum);
	}

	int inMarkerCount = 0;
	int outMarkerCount = 0;
	for (auto &marker : aMarkers) {
		graphData.setTWeights(marker.second - 1, !marker.first ? 10000.0f : 0.0f, marker.first ? 10000.0f : 0.0f);
		if (marker.first) {
			++inMarkerCount;
		} else {
			++outMarkerCount;
		}
	}
	BOOST_LOG_TRIVIAL(info) << boost::str(boost::format("Edge weights statistics: min %1%, max %2%, mean %3%, in markers %4%, out markers %5%")
		% ba::extract_result<ba::tag::min>(acc)
		% ba::extract_result<ba::tag::max>(acc)
		% ba::extract_result<ba::tag::mean>(acc)
		% inMarkerCount
		% outMarkerCount);
	BOOST_LOG_TRIVIAL(info) << "Computing max flow ...";
	std::vector<int> maskedObjectIds(vertexCount);
	computeCudaGraphCutImplementation(graphData, aConfig, maskedObjectIds);

	std::vector<int> markers;
	for (int i = 0; i < maskedObjectIds.size(); ++i) {
		if (maskedObjectIds[i]) {
			markers.push_back(i + 1);
		}
	}
	return markers;
}
