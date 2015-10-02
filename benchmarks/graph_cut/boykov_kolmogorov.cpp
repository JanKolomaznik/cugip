#include <BK301/graph.h>

void
computeBoykovKolmogorovGrid(
	float *aData,
	uint8_t *aMarkers,
	int aWidth,
	int aHeight,
	int aDepth)
{
	int edgeCount;
	int vertexCount;
	typedef Graph<float, float, float> GraphType;

	GraphType graph(vertexCount, edgeCount);

	graph.add_node(vertexCount);

	for (int k = 0; k < aDepth; ++k) {
		for (int j = 0; j < aHeight; ++j) {
			for (int i = 0; i < aWidth; ++i) {

			}
		}
	}
	float flow = graph.maxflow();

}
