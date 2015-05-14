#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/device_flag.hpp>
#include <limits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>


#include <boost/filesystem.hpp>
#include <fstream>

#include "graph_cut_data.hpp"

#include "graph_cut_push.hpp"
#include "graph_cut_relabeling.hpp"

#define INVALID_LABEL (1 << 30)

namespace cugip {

template<typename TFlow>
CUGIP_GLOBAL void
initResidualsKernel(TFlow *aWeightsForward, TFlow *aWeightsBackward, EdgeResidualsRecord<TFlow> *aResiduals, int aSize)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if (edgeIdx < aSize) {
		aResiduals[edgeIdx].residuals[0] = aWeightsForward[edgeIdx];
		aResiduals[edgeIdx].residuals[1] = aWeightsBackward[edgeIdx];
	}
}


struct GraphCutPolicy
{
	struct RelabelPolicy {};
	struct PushPolicy {};
};

template<typename TGraphData, typename TPolicy>
struct MinCut
{
	static float
	compute(TGraphData &aGraph)
	{
		//boost::timer::cpu_timer timer;
		//timer.start();
		//CUGIP_DPRINT("MAX FLOW");
		//init_residuals(aGraph);
		push_through_tlinks_from_source(aGraph);

		//debug_print();
		CUGIP_CHECK_ERROR_STATE("After max_flow init");
		//timer.stop();
		//std::cout << timer.format(9, "%w") << "\n";
		bool done = false;
		size_t iteration = 0;
		while(!done) {
			//timer.start();
			Relabel<TGraphData, typename TPolicy::RelabelPolicy>::compute(aGraph);
			//assign_label_by_distance();

			done = !Push<TGraphData, typename TPolicy::PushPolicy>::compute(aGraph);
			//done = !push();
			//timer.stop();
			//CUGIP_DPRINT("**iteration " << iteration << ": " << timer.format(9, "%w"));
			++iteration;
		}
		return computeFlowThroughSinkFrontier(aGraph);
	}

	static void
	push_through_tlinks_from_source(TGraphData &aGraph)
	{
		//CUGIP_DPRINT("push_through_tlinks_from_source");

		dim3 blockSize1D( 512 );
		dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

		pushThroughTLinksFromSourceKernel<<<gridSize1D, blockSize1D>>>(aGraph);

		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After pushThroughTLinksFromSourceKernel");
	}

	static void
	push_through_tlinks_to_sink(TGraphData &aGraph)
	{
		//CUGIP_DPRINT("push_through_tlinks_to_sink");

		dim3 blockSize1D( 512 );
		dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

		pushThroughTLinksToSinkKernel<<<gridSize1D, blockSize1D>>>(aGraph);

		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After push_through_tlinks_to_sink");
	}

	static float
	computeFlowThroughSinkFrontier(TGraphData &aGraph)
	{
		push_through_tlinks_to_sink(aGraph);
		return thrust::reduce(aGraph.mSinkFlow, aGraph.mSinkFlow + aGraph.vertexCount());
	}
};

template<typename TGraph>
class MinimalGraphCutComputation
{
public:
	typedef float Flow;
	MinimalGraphCutComputation()
		: mGraph(nullptr)
	{}

	void
	setGraph(TGraph &aGraph)
	{
		mGraph = &aGraph;
	}

	Flow
	run()
	{
		init_residuals();
		return MinCut<GraphCutData<Flow>, GraphCutPolicy>::compute(mGraphData);
	}

	void
	init_residuals()
	{
		//CUGIP_DPRINT("init_residuals()");
		dim3 blockSize1D( 512 );
		dim3 gridSize1D((mGraphData.mEdgeCount + blockSize1D.x - 1) / (blockSize1D.x), 1);

		initResidualsKernel<<<gridSize1D, blockSize1D>>>(
				thrust::raw_pointer_cast(mGraph->mEdgeWeightsForward.data()),
				thrust::raw_pointer_cast(mGraph->mEdgeWeightsBackward.data()),
				thrust::raw_pointer_cast(mGraphData.mResiduals),
				mGraphData.mEdgeCount
				);

		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After init_residuals()");
	}


protected:
	TGraph *mGraph;
	GraphCutData<Flow> mGraphData;
};




} // namespace cugip
