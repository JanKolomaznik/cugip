#include <GridCut/GridGraph_3D_6C.h>
#include <GridCut/GridGraph_3D_26C.h>

#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/detail/for_each_host.hpp>

#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>

void
computeGridCut(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const uint8_t, 3> aMarkers,
	cugip::host_image_view<uint8_t, 3> aOutput,
	float aSigma)
{
	using namespace cugip;

	VonNeumannNeighborhood<3> neighborhood;
	//MooreNeighborhood<3> neighborhood;
	//int edgeCount = product(aData.dimensions()) * neighborhood.size() / 2;
	int vertexCount = product(aData.dimensions());

	typedef GridGraph_3D_6C<float, float, float> GraphType;

	BOOST_LOG_TRIVIAL(info) << "Constructing graph output: vertices = " << vertexCount;
	GraphType graph(aData.dimensions()[0], aData.dimensions()[1], aData.dimensions()[2]);

	auto size = aData.dimensions();
	BOOST_LOG_TRIVIAL(info) << "Setting edge weights...";
	region<3> imageRegion{ Int3(), size };
	for_each(
		imageRegion,
		[&](const Int3 &coordinate) {
			int node = graph.node_id(coordinate[0], coordinate[1], coordinate[2]);
			float source_weight = aMarkers[coordinate] == 128 ? 1000000.f : 0.0f;
			float sink_weight = aMarkers[coordinate] == 255 ? 1000000.f : 0.0f;
			if (min(coordinate) <= 1 || min(size - coordinate) <= 2) {
				source_weight = 1000000.0f;
				sink_weight = 0.0f;
			}
			graph.set_terminal_cap(node, source_weight, sink_weight);
			for (int n = 1; n < neighborhood.size(); ++n) {
				Int3 neighbor = coordinate + neighborhood.offset(n);
				Int3 offset = neighborhood.offset(n);
				if (isInsideRegion(aData.dimensions(), neighbor)) {
					float weight =  std::exp(-sqr(aData[coordinate] - aData[neighbor]) / 2.0f * sqr(aSigma));
					graph.set_neighbor_cap(node, offset[0], offset[1], offset[2], weight);
				}
			}

		});
	BOOST_LOG_TRIVIAL(info) << "Computing max flow ...";
	boost::timer::cpu_timer computationTimer;
	computationTimer.start();
	graph.compute_maxflow();
	computationTimer.stop();
	BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");

	BOOST_LOG_TRIVIAL(info) << "Filling output ...";
	for_each(
		imageRegion,
		[&](const Int3 &coordinate) {
			int node = graph.node_id(coordinate[0], coordinate[1], coordinate[2]);
			aOutput[coordinate] = graph.get_segment(node) ? 255 : 0;
		});
}
