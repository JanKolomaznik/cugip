#include <cugip/advanced_operations/graph_cut.hpp>
#include <cugip/advanced_operations/detail/graph_cut_host_utils.hpp>

#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/detail/for_each_host.hpp>
#include <cugip/access_utils.hpp>

#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>


void
computeCudaGraphCutImplementation(const cugip::GraphData<float> &aGraphData, cugip::host_image_view<uint8_t, 3> aOutput, uint8_t aMaskValue)
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
	float flow = graph.max_flow();
	computationTimer.stop();
	BOOST_LOG_TRIVIAL(info) << "Max flow: " << flow;
	//BOOST_LOG_TRIVIAL(info) << "Computation time: " << computationTimer.format(9, "%w");
	//BOOST_LOG_TRIVIAL(info) << "Computation time: " << boost::timer::format(computationTimer.elapsed(), 9, "%w");
	BOOST_LOG_TRIVIAL(info) << "Computation time: " << (computationTimer.elapsed().wall / 1000000000.0f);
	BOOST_LOG_TRIVIAL(info) << "Filling output ...";
	graph.fill_segments(linear_access_view(aOutput), aMaskValue, 0);


}
