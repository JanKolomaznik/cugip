#include <cugip/advanced_operations/graph_cut.hpp>


void
computeCudaGraphCut(
	float *aData,
	uint8_t *aMarkers,
	int aWidth,
	int aHeight,
	int aDepth)
{
	int edgeCount;
	int vertexCount;

	std::vector<float> tlinksSource(vertexCount);
	std::vector<float> tlinksSink(vertexCount);
	std::vector<cugip::EdgeRecord> edges(edgeCount);
	std::vector<float> weights(edgeCount);
	std::vector<float> weightsBackward(edgeCount);

	typedef Graph<float, float, float> GraphType;

	GraphType graph(vertexCount, edgeCount);

	graph.add_node(vertexCount);

	for (int k = 0; k < aDepth; ++k) {
		for (int j = 0; j < aHeight; ++j) {
			for (int i = 0; i < aWidth; ++i) {

			}
		}
	}

	cugip::Graph<float> graph;
	graph.set_vertex_count(w*h);

	graph.set_nweights(
		edges.size(),
		&(edges[0]),
		&(weights[0]),
		&(weightsBackward[0]));

	graph.set_tweights(
		&(tlinksSource[0]),
		&(tlinksSink[0])
		);

	float flow = graph.maxflow();

}
