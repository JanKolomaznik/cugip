#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

#include <cstdio>
#include <cstdlib>

//#include <graph.h>

#include <cugip/advanced_operations/graph_cut.hpp>
#include <cugip/exception.hpp>

#include <vector>


#include "dataset.h"
#include "mfi.h"
#include "timer.h"

#include <cuda.h>
#include <cudaProfiler.h>

class CudaDeleter
{
public:
	CudaDeleter() { }
	~CudaDeleter()
	{
		CUGIP_CHECK_ERROR_STATE("Before CudaDeleter");
		printf("CudaDeleter...\n");
		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		//cudaProfilerStop();
		cudaDeviceReset();
		printf("...done\n");
	}
};
//static CudaDeleter cleanerUpper;

double t1;
double t2;

#define CLOCK_START()    { timerReset();    t1 = timerGet(); }
#define CLOCK_STOP(TIME) { t2 = timerGet(); *TIME = t2-t1;   }

void run_test(double* time_init,double* time_maxflow,double* time_output)
{
	const int w = 1000;
	const int h = 1000;
	std::vector<float> tlinksSource(w*h);
	std::vector<float> tlinksSink(w*h);
	std::vector<cugip::EdgeRecord> edges((w-1)*h + (h-1)*w);
	std::vector<float> weights((w-1)*h + (h-1)*w);
	std::vector<float> weightsBackward((w-1)*h + (h-1)*w);

	for(int y=0;y<h;y++) {
		for(int x=0;x<w;x++) {
			tlinksSource[x+y*w] = (y == 0) ? 100000 : 0;
			tlinksSink[x+y*w] = (y == h - 1) ? 100000 : 0;
			if (x<w-1) {
				edges[x+y*(w-1)] = cugip::EdgeRecord(x+y*w, (x+1)+y*w);
				weights[x+y*(w-1)] = 50;
				weightsBackward[x+y*(w-1)] = 50;
			}
			if (y<h-1) {
				edges[(w-1)*(h) + x+y*(w-1)] = cugip::EdgeRecord(x+y*w,x+(y+1)*w);
				weights[(w-1)*(h) + x+y*(w-1)] = std::abs(y - 50) + 1;
				weightsBackward[(w-1)*(h) + x+y*(w-1)] = std::abs(y - 50) + 1;
			}
		}
	}
	CLOCK_START();
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

	CLOCK_STOP(time_init);

	CLOCK_START();
	graph.max_flow();
	CLOCK_STOP(time_maxflow);

	CLOCK_START();
	CLOCK_STOP(time_output);
}


template<typename type_terminal_cap,typename type_neighbor_cap>
void run_BK301_2D_4C(MFI* mfi,unsigned char* out_label,int* out_maxflow,double* time_init,double* time_maxflow,double* time_output)
{
	printf("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbba\n");
	const int w = mfi->width;
	const int h = mfi->height;
	//printf("width %d, height %d\n", w, h);
	const type_terminal_cap* cap_source = (type_terminal_cap*)mfi->cap_source;
	const type_terminal_cap* cap_sink   = (type_terminal_cap*)mfi->cap_sink;

	const type_neighbor_cap* cap_neighbor[4] = { (type_neighbor_cap*)(mfi->cap_neighbor[0]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[1]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[2]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[3]) };
	std::vector<float> tlinksSource(w*h);
	std::vector<float> tlinksSink(w*h);
	std::vector<cugip::EdgeRecord> edges((w-1)*h + (h-1)*w);
	std::vector<float> weights((w-1)*h + (h-1)*w);
	std::vector<float> weightsBackward((w-1)*h + (h-1)*w);

	int source_count = 0;
	int sink_count = 0;
	int lastEdge = 0;
	for(int y=0;y<h;y++) {
		for(int x=0;x<w;x++) {
			if (cap_source[x+y*w] > 0) {
				++source_count;
			}
			if (cap_sink[x+y*w] > 0) {
				++sink_count;
			}
			tlinksSource[x+y*w] = cap_source[x+y*w];
			tlinksSink[x+y*w] = cap_sink[x+y*w];
			if (x<w-1) {
				edges[lastEdge/*x+y*(w-1)*/] = cugip::EdgeRecord(x+y*w, (x+1)+y*w);
				weights[lastEdge/*x+y*(w-1)*/] = cap_neighbor[MFI::ARC_LE][x+1+y*w];
				weightsBackward[lastEdge/*x+y*(w-1)*/] = cap_neighbor[MFI::ARC_GE][x+y*w];
				if (weights[lastEdge] > 0.0f && weightsBackward[lastEdge] > 0.0f) {
					++lastEdge;
				}
			}
			if (y<h-1) {
				edges[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cugip::EdgeRecord(x+y*w,x+(y+1)*w);
				weights[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cap_neighbor[MFI::ARC_EL][x+(y+1)*w];
				weightsBackward[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cap_neighbor[MFI::ARC_EG][x+y*w];
				if (weights[lastEdge] > 0.0f && weightsBackward[lastEdge] > 0.0f) {
					++lastEdge;
				}
			}
		}
	}
	CLOCK_START();
	cugip::Graph<float> graph;
	graph.set_vertex_count(w*h);

	printf("Edge count %d %d\n", lastEdge, edges.size());
	graph.set_nweights(
		lastEdge,//edges.size(),
		&(edges[0]),
		&(weights[0]),
		&(weightsBackward[0]));

	graph.set_tweights(
		&(tlinksSource[0]),
		&(tlinksSink[0])
		);


	CLOCK_STOP(time_init);

	CLOCK_START();
	*out_maxflow = graph.max_flow();
	CLOCK_STOP(time_maxflow);

	CLOCK_START();
	//for(int xy=0;xy<w*h;xy++) out_label[xy] = graph->what_segment(xy);

	//delete graph;
	CLOCK_STOP(time_output);
}

template<typename type_terminal_cap,typename type_neighbor_cap>
void run_BK301_3D_6C(MFI* mfi,unsigned char* out_label,int* out_maxflow,double* time_init,double* time_maxflow,double* time_output)
{
	printf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");
	const int w = mfi->width;
	const int h = mfi->height;
	const int d = mfi->depth;

	const type_terminal_cap* cap_source = (type_terminal_cap*)mfi->cap_source;
	const type_terminal_cap* cap_sink   = (type_terminal_cap*)mfi->cap_sink;

	const type_neighbor_cap* cap_neighbor[6] = { (type_neighbor_cap*)(mfi->cap_neighbor[0]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[1]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[2]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[3]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[4]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[5]) };

	const int num_nodes = w*h*d;
	const int num_edges = (w-1)*(h*d) + (h-1)*(w*d) + (d-1)*(w*h);

	std::vector<float> tlinksSource(num_nodes);
	std::vector<float> tlinksSink(num_nodes);
	std::vector<cugip::EdgeRecord> edges(num_edges);
	std::vector<float> weights(num_edges);
	std::vector<float> weightsBackward(num_edges);

	int source_count = 0;
	int sink_count = 0;
	int lastEdge = 0;
	for(int z=0;z<d;z++) {
		for(int y=0;y<h;y++) {
			for(int x=0;x<w;x++) {
				int vertex = x+y*w+z*(w*h);
				if (cap_source[vertex] > 0) {
					++source_count;
				}
				if (cap_sink[vertex] > 0) {
					++sink_count;
				}
				tlinksSource[vertex] = cap_source[vertex];
				tlinksSink[vertex] = cap_sink[vertex];
				if (x<w-1) {
					edges[lastEdge/*x+y*(w-1)*/] = cugip::EdgeRecord(vertex, vertex + 1);
					weights[lastEdge/*x+y*(w-1)*/] = cap_neighbor[MFI::ARC_LEE][vertex + 1];
					weightsBackward[lastEdge/*x+y*(w-1)*/] = cap_neighbor[MFI::ARC_GEE][vertex];
					if (weights[lastEdge] > 0.0f && weightsBackward[lastEdge] > 0.0f) {
						++lastEdge;
					}
				}
				if (y<h-1) {
					edges[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cugip::EdgeRecord(vertex,vertex + w);
					weights[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cap_neighbor[MFI::ARC_ELE][vertex + w];
					weightsBackward[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cap_neighbor[MFI::ARC_EGE][vertex];
					if (weights[lastEdge] > 0.0f && weightsBackward[lastEdge] > 0.0f) {
						++lastEdge;
					}
				}
				if (z<h-1) {
					edges[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cugip::EdgeRecord(vertex, vertex + h*w);
					weights[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cap_neighbor[MFI::ARC_EEL][vertex + h*w];
					weightsBackward[lastEdge/*(w-1)*(h) + x+y*(w-1)*/] = cap_neighbor[MFI::ARC_EEG][vertex];
					if (weights[lastEdge] > 0.0f && weightsBackward[lastEdge] > 0.0f) {
						++lastEdge;
					}
				}
    //if (z<d-1) graph->add_edge(x+y*w+z*(w*h),x+y*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_EEG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_EEL][x+y*w+(z+1)*(w*h)]);
			}
		}
	}
	CLOCK_START();
	cugip::Graph<float> graph;
	graph.set_vertex_count(num_nodes);

	printf("Edge count %d %d\n", lastEdge, edges.size());
	graph.set_nweights(
		lastEdge,//edges.size(),
		&(edges[0]),
		&(weights[0]),
		&(weightsBackward[0]));

	graph.set_tweights(
		&(tlinksSource[0]),
		&(tlinksSink[0])
		);


	CLOCK_STOP(time_init);

	CLOCK_START();
	*out_maxflow = graph.max_flow();
	CLOCK_STOP(time_maxflow);

	CLOCK_START();
	//for(int xy=0;xy<w*h;xy++) out_label[xy] = graph->what_segment(xy);

	//delete graph;
	CLOCK_STOP(time_output);
  /*const int w = mfi->width;
  const int h = mfi->height;
  const int d = mfi->depth;

  const type_terminal_cap* cap_source = (type_terminal_cap*)mfi->cap_source;
  const type_terminal_cap* cap_sink   = (type_terminal_cap*)mfi->cap_sink;

  const type_neighbor_cap* cap_neighbor[6] = { (type_neighbor_cap*)(mfi->cap_neighbor[0]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[1]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[2]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[3]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[4]),
					       (type_neighbor_cap*)(mfi->cap_neighbor[5]) };

  const int num_nodes = w*h*d;
  const int num_edges = (w-1)*(h*d) + (h-1)*(w*d) + (d-1)*(w*h);

  typedef Graph<int,int,int> GraphType;

  CLOCK_START();
  GraphType* graph = new GraphType(num_nodes,num_edges);

  graph->add_node(num_nodes);

  for(int z=0;z<d;z++)
  for(int y=0;y<h;y++)
  for(int x=0;x<w;x++)
  {
    graph->add_tweights(x+y*w+z*(w*h),cap_source[x+y*w+z*(w*h)],cap_sink[x+y*w+z*(w*h)]);

    if (x<w-1) graph->add_edge(x+y*w+z*(w*h),(x+1)+y*w+z*(w*h),cap_neighbor[MFI::ARC_GEE][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LEE][(x+1)+y*w+z*(w*h)]);
    if (y<h-1) graph->add_edge(x+y*w+z*(w*h),x+(y+1)*w+z*(w*h),cap_neighbor[MFI::ARC_EGE][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_ELE][x+(y+1)*w+z*(w*h)]);
    if (z<d-1) graph->add_edge(x+y*w+z*(w*h),x+y*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_EEG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_EEL][x+y*w+(z+1)*(w*h)]);
  }
  CLOCK_STOP(time_init);

  CLOCK_START();
  *out_maxflow = graph->maxflow();
  CLOCK_STOP(time_maxflow);

  CLOCK_START();
  for(int xyz=0;xyz<num_nodes;xyz++) out_label[xyz] = graph->what_segment(xyz);

  delete graph;
  CLOCK_STOP(time_output);*/
}

template<typename type_terminal_cap,typename type_neighbor_cap>
void run_BK301_3D_26C(MFI* mfi,unsigned char* out_label,int* out_maxflow,double* time_init,double* time_maxflow,double* time_output)
{
	printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxa\n");
  /*const int w = mfi->width;
  const int h = mfi->height;
  const int d = mfi->depth;

  const type_terminal_cap* cap_source = (type_terminal_cap*)mfi->cap_source;
  const type_terminal_cap* cap_sink   = (type_terminal_cap*)mfi->cap_sink;

  const type_neighbor_cap* cap_neighbor[26] = { (type_neighbor_cap*)(mfi->cap_neighbor[ 0]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 1]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 2]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 3]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 4]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 5]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 6]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 7]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 8]),
						(type_neighbor_cap*)(mfi->cap_neighbor[ 9]),
						(type_neighbor_cap*)(mfi->cap_neighbor[10]),
						(type_neighbor_cap*)(mfi->cap_neighbor[11]),
						(type_neighbor_cap*)(mfi->cap_neighbor[12]),
						(type_neighbor_cap*)(mfi->cap_neighbor[13]),
						(type_neighbor_cap*)(mfi->cap_neighbor[14]),
						(type_neighbor_cap*)(mfi->cap_neighbor[15]),
						(type_neighbor_cap*)(mfi->cap_neighbor[16]),
						(type_neighbor_cap*)(mfi->cap_neighbor[17]),
						(type_neighbor_cap*)(mfi->cap_neighbor[18]),
						(type_neighbor_cap*)(mfi->cap_neighbor[19]),
						(type_neighbor_cap*)(mfi->cap_neighbor[20]),
						(type_neighbor_cap*)(mfi->cap_neighbor[21]),
						(type_neighbor_cap*)(mfi->cap_neighbor[22]),
						(type_neighbor_cap*)(mfi->cap_neighbor[23]),
						(type_neighbor_cap*)(mfi->cap_neighbor[24]),
						(type_neighbor_cap*)(mfi->cap_neighbor[25]) };

  const int num_nodes = w*h*d;
  const int num_edges = 13*w*h*d;

  typedef Graph<int,int,int> GraphType;

  CLOCK_START();
  GraphType* graph = new GraphType(num_nodes,num_edges);

  graph->add_node(num_nodes);

  for(int z=0;z<d;z++)
  for(int y=0;y<h;y++)
  for(int x=0;x<w;x++)
  {
    graph->add_tweights(x+y*w+z*(w*h),cap_source[x+y*w+z*(w*h)],cap_sink[x+y*w+z*(w*h)]);

    if (x<w-1 && y<h-1 && z<d-1) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y+1)*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_GGG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LLL][(x+1)+(y+1)*w+(z+1)*(w*h)]);
    if (x<w-1          && z<d-1) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y  )*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_GEG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LEL][(x+1)+(y  )*w+(z+1)*(w*h)]);
    if (x<w-1 && y>0   && z<d-1) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y-1)*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_GLG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LGL][(x+1)+(y-1)*w+(z+1)*(w*h)]);
    if (x<w-1 && y<h-1         ) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y+1)*w+(z  )*(w*h),cap_neighbor[MFI::ARC_GGE][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LLE][(x+1)+(y+1)*w+(z  )*(w*h)]);
    if (x<w-1                  ) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y  )*w+(z  )*(w*h),cap_neighbor[MFI::ARC_GEE][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LEE][(x+1)+(y  )*w+(z  )*(w*h)]);
    if (x<w-1 && y>0           ) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y-1)*w+(z  )*(w*h),cap_neighbor[MFI::ARC_GLE][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LGE][(x+1)+(y-1)*w+(z  )*(w*h)]);
    if (x<w-1 && y<h-1 && z>0  ) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y+1)*w+(z-1)*(w*h),cap_neighbor[MFI::ARC_GGL][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LLG][(x+1)+(y+1)*w+(z-1)*(w*h)]);
    if (x<w-1          && z>0  ) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y  )*w+(z-1)*(w*h),cap_neighbor[MFI::ARC_GEL][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LEG][(x+1)+(y  )*w+(z-1)*(w*h)]);
    if (x<w-1 && y>0   && z>0  ) graph->add_edge(x+y*w+z*(w*h),(x+1)+(y-1)*w+(z-1)*(w*h),cap_neighbor[MFI::ARC_GLL][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_LGG][(x+1)+(y-1)*w+(z-1)*(w*h)]);
    if (         y<h-1 && z<d-1) graph->add_edge(x+y*w+z*(w*h),(x  )+(y+1)*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_EGG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_ELL][(x  )+(y+1)*w+(z+1)*(w*h)]);
    if (                  z<d-1) graph->add_edge(x+y*w+z*(w*h),(x  )+(y  )*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_EEG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_EEL][(x  )+(y  )*w+(z+1)*(w*h)]);
    if (         y>0   && z<d-1) graph->add_edge(x+y*w+z*(w*h),(x  )+(y-1)*w+(z+1)*(w*h),cap_neighbor[MFI::ARC_ELG][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_EGL][(x  )+(y-1)*w+(z+1)*(w*h)]);
    if (         y<h-1         ) graph->add_edge(x+y*w+z*(w*h),(x  )+(y+1)*w+(z  )*(w*h),cap_neighbor[MFI::ARC_EGE][x+y*w+z*(w*h)],cap_neighbor[MFI::ARC_ELE][(x  )+(y+1)*w+(z  )*(w*h)]);
  }
  CLOCK_STOP(time_init);

  CLOCK_START();
  *out_maxflow = graph->maxflow();
  CLOCK_STOP(time_maxflow);

  CLOCK_START()
  for(int xyz=0;xyz<num_nodes;xyz++) out_label[xyz] = graph->what_segment(xyz);

  delete graph;
  CLOCK_STOP(time_output);*/
}

int run(const char *dataset_path)
{
	CudaDeleter cleanerUpper;
	cudaDeviceReset();
	std::cout << cugip::cudaDeviceInfoText();


	void (*run_BK301[4][27][3][3])(MFI*,unsigned char*,int*,double*,double*,double*);

	run_BK301[2][ 4][MFI::TYPE_UINT8 ][MFI::TYPE_UINT8 ] = run_BK301_2D_4C<unsigned char ,unsigned char >;
	run_BK301[2][ 4][MFI::TYPE_UINT16][MFI::TYPE_UINT8 ] = run_BK301_2D_4C<unsigned short,unsigned char >;
	run_BK301[2][ 4][MFI::TYPE_UINT32][MFI::TYPE_UINT8 ] = run_BK301_2D_4C<unsigned int,  unsigned char >;
	run_BK301[2][ 4][MFI::TYPE_UINT8 ][MFI::TYPE_UINT16] = run_BK301_2D_4C<unsigned char ,unsigned short>;
	run_BK301[2][ 4][MFI::TYPE_UINT16][MFI::TYPE_UINT16] = run_BK301_2D_4C<unsigned short,unsigned short>;
	run_BK301[2][ 4][MFI::TYPE_UINT32][MFI::TYPE_UINT16] = run_BK301_2D_4C<unsigned int,  unsigned short>;
	run_BK301[2][ 4][MFI::TYPE_UINT8 ][MFI::TYPE_UINT32] = run_BK301_2D_4C<unsigned char ,unsigned int  >;
	run_BK301[2][ 4][MFI::TYPE_UINT16][MFI::TYPE_UINT32] = run_BK301_2D_4C<unsigned short,unsigned int  >;
	run_BK301[2][ 4][MFI::TYPE_UINT32][MFI::TYPE_UINT32] = run_BK301_2D_4C<unsigned int,  unsigned int  >;
	run_BK301[3][ 6][MFI::TYPE_UINT8 ][MFI::TYPE_UINT8 ] = run_BK301_3D_6C<unsigned char ,unsigned char >;
	run_BK301[3][ 6][MFI::TYPE_UINT16][MFI::TYPE_UINT8 ] = run_BK301_3D_6C<unsigned short,unsigned char >;
	run_BK301[3][ 6][MFI::TYPE_UINT32][MFI::TYPE_UINT8 ] = run_BK301_3D_6C<unsigned int,  unsigned char >;
	run_BK301[3][ 6][MFI::TYPE_UINT8 ][MFI::TYPE_UINT16] = run_BK301_3D_6C<unsigned char ,unsigned short>;
	run_BK301[3][ 6][MFI::TYPE_UINT16][MFI::TYPE_UINT16] = run_BK301_3D_6C<unsigned short,unsigned short>;
	run_BK301[3][ 6][MFI::TYPE_UINT32][MFI::TYPE_UINT16] = run_BK301_3D_6C<unsigned int,  unsigned short>;
	run_BK301[3][ 6][MFI::TYPE_UINT8 ][MFI::TYPE_UINT32] = run_BK301_3D_6C<unsigned char ,unsigned int  >;
	run_BK301[3][ 6][MFI::TYPE_UINT16][MFI::TYPE_UINT32] = run_BK301_3D_6C<unsigned short,unsigned int  >;
	run_BK301[3][ 6][MFI::TYPE_UINT32][MFI::TYPE_UINT32] = run_BK301_3D_6C<unsigned int,  unsigned int  >;
	run_BK301[3][26][MFI::TYPE_UINT8 ][MFI::TYPE_UINT8 ] = run_BK301_3D_26C<unsigned char ,unsigned char >;
	run_BK301[3][26][MFI::TYPE_UINT16][MFI::TYPE_UINT8 ] = run_BK301_3D_26C<unsigned short,unsigned char >;
	run_BK301[3][26][MFI::TYPE_UINT32][MFI::TYPE_UINT8 ] = run_BK301_3D_26C<unsigned int,  unsigned char >;
	run_BK301[3][26][MFI::TYPE_UINT8 ][MFI::TYPE_UINT16] = run_BK301_3D_26C<unsigned char ,unsigned short>;
	run_BK301[3][26][MFI::TYPE_UINT16][MFI::TYPE_UINT16] = run_BK301_3D_26C<unsigned short,unsigned short>;
	run_BK301[3][26][MFI::TYPE_UINT32][MFI::TYPE_UINT16] = run_BK301_3D_26C<unsigned int,  unsigned short>;
	run_BK301[3][26][MFI::TYPE_UINT8 ][MFI::TYPE_UINT32] = run_BK301_3D_26C<unsigned char ,unsigned int  >;
	run_BK301[3][26][MFI::TYPE_UINT16][MFI::TYPE_UINT32] = run_BK301_3D_26C<unsigned short,unsigned int  >;
	run_BK301[3][26][MFI::TYPE_UINT32][MFI::TYPE_UINT32] = run_BK301_3D_26C<unsigned int,  unsigned int  >;

	//  const char* dataset_path = argc==2 ? argv[1] : "./dataset";

	int num_instances = (sizeof(instances)/sizeof(Instance));
	//num_instances = 0;

	printf("instance                            time-init  time-maxflow  time-output  total\n");

	/*{
		double sum_time_init = 0.0;
		double sum_time_maxflow = 0.0;
		double sum_time_output = 0.0;

		run_test(&sum_time_init, &sum_time_maxflow, &sum_time_output);
		double sum_time_total = sum_time_init + sum_time_maxflow + sum_time_output;
		printf("%-38s % 6.0f        % 6.0f       % 6.0f % 6.0f\n",
		       "TEST",sum_time_init,sum_time_maxflow,sum_time_output,sum_time_total);
	}*/
	for(int i=0;i<num_instances;i++)
	{
		double sum_time_init = 0.0;
		double sum_time_maxflow = 0.0;
		double sum_time_output = 0.0;

		for(int j=0;j<instances[i].count;j++)
		{
			char filename[1024];

			if (instances[i].count==1)
			{
				sprintf(filename,instances[i].filename,dataset_path);
			}
			else
			{
				sprintf(filename,instances[i].filename,dataset_path,j);
			}

			MFI* mfi = mfi_read(filename);

			if (!mfi)
			{
				printf("FAILED to read instance %s\n",filename);
				return 1;
			}

			unsigned char* label = (unsigned char*)malloc(mfi->width*mfi->height*mfi->depth);

			int maxflow = -1;

			double time_init;
			double time_maxflow;
			double time_output;

			run_BK301[mfi->dimension]
					[mfi->connectivity]
					[mfi->type_terminal_cap]
					[mfi->type_neighbor_cap](mfi,label,&maxflow,&time_init,&time_maxflow,&time_output);

			sum_time_init    += time_init;
			sum_time_maxflow += time_maxflow;
			sum_time_output  += time_output;
			printf("Flow: %d - %d\n", maxflow, (int)(mfi->maxflow));
			/*if (maxflow != mfi->maxflow)
      {
	printf("INVALID maxflow value returned for instance %s\n",filename);
	return 1;
      }*/

			free(label);

			mfi_free(mfi);
		}

		double sum_time_total = sum_time_init + sum_time_maxflow + sum_time_output;

		printf("%-38s % 6.0f        % 6.0f       % 6.0f % 6.0f\n",
		       instances[i].name,sum_time_init,sum_time_maxflow,sum_time_output,sum_time_total);
	}

	return 0;
}

