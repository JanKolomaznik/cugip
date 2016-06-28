#include <cstdint>

#include <BK301/graph.h>

#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/detail/for_each_host.hpp>

#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>

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
	BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");

	BOOST_LOG_TRIVIAL(info) << "Filling output ...";
	for_each(
		imageRegion,
		[&](const Int3 &coordinate) {
			aOutput[coordinate] = graph.what_segment(get_linear_access_index(size, coordinate)) == GraphType::SINK ? aMaskValue : 0;
		});
}
