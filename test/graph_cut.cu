#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__


#include <cugip/advanced_operations/graph_cut.hpp>

#include <vector>


void
test_graph_cut()
{
	using namespace cugip;
	cugip::Graph<float> graph;
	graph.set_vertex_count(16);

	//std::vector<int>
	int nlinksVertices1[24] = {
		0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
		};
	//std::vector<int>
	int nlinksVertices2[24] = {
		1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 
		4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
		};

	EdgeRecord edges[24];
	for (int i = 0; i < 24; ++i) {
		edges[i] = EdgeRecord(nlinksVertices1[i], nlinksVertices2[i]);
	}

	//std::vector<float> 
	float nlinksWeights[24] = {
		5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 
		5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f
		};

	float tlinksSource[16] = {
		100.0f, 100.0f, 100.0f, 100.0f, 
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
		};

	float tlinksSink[16] = {
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		100.0f, 100.0f, 100.0f, 100.0f
		};

	graph.set_nweights(
		24,
		/*nlinksVertices1,
		nlinksVertices2,*/
		edges,
		nlinksWeights, 
		nlinksWeights);

	graph.set_tweights(
		tlinksSource,
		tlinksSink
		);

	float flow = graph.max_flow();
	CUGIP_DPRINT("Max flow = " << flow);
}

