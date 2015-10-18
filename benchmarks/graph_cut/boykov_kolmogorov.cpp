#include <cstdint>

#include <BK301/graph.h>

#include <cugip/host_image_view.hpp>

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

	auto data = cugip::makeConstHostImageView(aData, cugip::Int3(aWidth, aHeight, aDepth));
	auto markers = cugip::makeConstHostImageView(aMarkers, cugip::Int3(aWidth, aHeight, aDepth));

	typedef Graph<float, float, float> GraphType;

	GraphType graph(vertexCount, edgeCount);

	graph.add_node(vertexCount);

	for (int k = 0; k < aDepth; ++k) {
		for (int j = 0; j < aHeight; ++j) {
			for (int i = 0; i < aWidth; ++i) {
				//graph.add_tweights(x+y*w,cap_source[x+y*w],cap_sink[x+y*w]);
				//if (x<w-1) graph.add_edge(x+y*w,(x+1)+y*w,cap_neighbor[MFI::ARC_GE][x+y*w],cap_neighbor[MFI::ARC_LE][(x+1)+y*w]);
			}
		}
	}
	float flow = graph.maxflow();

}
