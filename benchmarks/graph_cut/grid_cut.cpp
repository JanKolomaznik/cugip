#include <GridCut/GridGraph_3D_6C.h>
#include <GridCut/GridGraph_3D_26C.h>

void
computeGridCut(
	float *aData,
	uint8_t *aMarkers,
	int aWidth,
	int aHeight,
	int aDepth)
{
	int edgeCount;
	int vertexCount;
	typedef GridGraph_3D_6C<float, float, float> GraphType;
	GraphType graph(aWidth, aHeight, aDepth);

	graph.add_node(vertexCount);

	for (int k = 0; k < aDepth; ++k) {
		for (int j = 0; j < aHeight; ++j) {
			for (int i = 0; i < aWidth; ++i) {
				int node = graph.node_id(i, j, k);
				graph.set_terminal_cap(node,);
			}
		}
	}
	graph.compute_maxflow();
	float flow = graph.get_flow();
}
