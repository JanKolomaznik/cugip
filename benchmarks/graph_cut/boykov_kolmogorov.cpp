#include <cstdint>

#include <BK301/graph.h>

#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/for_each.hpp>

#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>
#include "graph.hpp"

static void printErrorMessage(char * aMessage) {
	BOOST_LOG_TRIVIAL(error) << aMessage;
	throw std::runtime_error(aMessage);
}

void
computeBoykovKolmogorovGrid(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const uint8_t, 3> aMarkers,
	cugip::host_image_view<uint8_t, 3> aOutput,
	float aSigma,
	uint8_t aMaskValue)
{
	using namespace cugip;

	VonNeumannNeighborhood<3> neighborhood;
	//MooreNeighborhood<3> neighborhood;
	int edgeCount = product(aData.dimensions()) * neighborhood.size() / 2;
	int vertexCount = product(aData.dimensions());

	typedef Graph<float, float, float> GraphType;

	BOOST_LOG_TRIVIAL(info) << "Constructing graph output: vertices = " << vertexCount << "; edges = " << edgeCount;
	GraphType graph(vertexCount, edgeCount, &printErrorMessage);

	BOOST_LOG_TRIVIAL(info) << "Setting node count: " << vertexCount;
	graph.add_node(vertexCount);
	auto size = aData.dimensions();
	BOOST_LOG_TRIVIAL(info) << "Setting edge weights...";
	region<3> imageRegion{ Int3(), size };
	for_each(
		imageRegion,
		[&](const Int3 &coordinate) {
			int centerIdx = get_linear_access_index(size, coordinate);
			float source_weight = aMarkers[coordinate] == 128 ? 1000000.f : 0.0f;
			float sink_weight = aMarkers[coordinate] == 255 ? 1000000.f : 0.0f;
			if (min(coordinate) <= 1 || min(size - coordinate) <= 2) {
				source_weight = 1000000.0f;
				sink_weight = 0.0f;
			}
			graph.add_tweights(centerIdx, source_weight, sink_weight);
			for (int n = 1; n < (neighborhood.size() + 1) / 2; ++n) {
				Int3 neighbor = coordinate + neighborhood.offset(n);
				int neighborIdx = get_linear_access_index(size, neighbor);
				if (isInsideRegion(aData.dimensions(), neighbor)) {
					float weight =  std::exp(-sqr(aData[coordinate] - aData[neighbor]) / 2.0f * sqr(aSigma));
					graph.add_edge(
						centerIdx,
						neighborIdx,
						weight,
						weight);
				}
			}

		});
	BOOST_LOG_TRIVIAL(info) << "Computing max flow ...";
	boost::timer::cpu_timer computationTimer;
	computationTimer.start();
	float flow = graph.maxflow();
	computationTimer.stop();
	BOOST_LOG_TRIVIAL(info) << "Max flow: " << flow;
	BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");

	BOOST_LOG_TRIVIAL(info) << "Filling output ...";
	for_each(
		imageRegion,
		[&](const Int3 &coordinate) {
			aOutput[coordinate] = graph.what_segment(get_linear_access_index(size, coordinate)) == GraphType::SINK ? aMaskValue : 0;
		});
}

std::vector<int>
computeBoykovKolmogorov(
	const GraphStats &aGraph,
	const std::vector<std::pair<bool, int>> &aMarkers)
{
	typedef Graph<float, float, float> GraphType;
	int edgeCount = aGraph.edges.size();//product(aData.dimensions()) * neighborhood.size() / 2;
	int vertexCount = aGraph.nodes.size();// product(aData.dimensions());
	BOOST_LOG_TRIVIAL(info) << "Edge count " << edgeCount;
	BOOST_LOG_TRIVIAL(info) << "Vertex count " << vertexCount;
	GraphType graph(vertexCount, edgeCount, &printErrorMessage);

	graph.add_node(vertexCount);

	BOOST_LOG_TRIVIAL(info) << "Node count = " << graph.get_node_num();
	BOOST_LOG_TRIVIAL(info) << "Adding edges ...";
	for (const auto &edge : aGraph.edges) {
		if (edge.first[0] > vertexCount || edge.first[1] > vertexCount ||
		edge.first[0] <= 0 || edge.first[1] <= 0) {
			BOOST_LOG_TRIVIAL(info) << "Wrong edge id: " << edge.first[0] << "; " << edge.first[1];
			throw "error";
		}
		//BOOST_LOG_TRIVIAL(info) << edge.first[0] << "; " << edge.first[1];
		graph.add_edge(
			edge.first[0]-1,
			edge.first[1]-1,
			edge.second.sum,
			edge.second.sum);
	}

	int inMarkerCount = 0;
	int outMarkerCount = 0;
	for (auto &marker : aMarkers) {
		graph.add_tweights(marker.second - 1, !marker.first ? 10000.0f : 0.0f, marker.first ? 10000.0f : 0.0f);
		if (marker.first) {
			++inMarkerCount;
		} else {
			++outMarkerCount;
		}
	}
	BOOST_LOG_TRIVIAL(info) << "in marker count " << inMarkerCount << ", out marker count " << outMarkerCount;
	BOOST_LOG_TRIVIAL(info) << "Computing max flow ...";
	boost::timer::cpu_timer computationTimer;
	computationTimer.start();
	float flow = graph.maxflow();
	computationTimer.stop();
	BOOST_LOG_TRIVIAL(info) << "Max flow: " << flow;
	BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");

	std::vector<int> markers;
	for (int i = 0; i < vertexCount; ++i) {
		if (graph.what_segment(i) == GraphType::SINK) {
			markers.push_back(i + 1);
		}
	}
	return markers;
}
