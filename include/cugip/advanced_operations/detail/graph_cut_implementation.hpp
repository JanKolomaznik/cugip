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
#include "graph_cut_policies.hpp"
#include <cuda_profiler_api.h>

//#define INVALID_LABEL (1 << 30)

namespace cugip {

template<typename TFlow>
CUGIP_GLOBAL void
initResidualsKernel(TFlow *aWeightsForward, TFlow *aWeightsBackward, EdgeResidualsRecord<TFlow> *aResiduals, int aSize)
{
	//uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

	while (edgeIdx < aSize) {
		aResiduals[edgeIdx].residuals[0] = aWeightsForward[edgeIdx];
		aResiduals[edgeIdx].residuals[1] = aWeightsBackward[edgeIdx];
		edgeIdx += gridDim.x * blockDim.x;
	}
}

template<typename TGraph, typename TGraphCutData>
void
graphCutDataFromGraph(
	TGraph &aGraph,
	TGraphCutData &aGraphData)
{
		aGraphData.vertexExcess = thrust::raw_pointer_cast(aGraph.mExcess.data()); // n
		aGraphData.labels = thrust::raw_pointer_cast(aGraph.mLabels.data());; // n
		//aGraphData.mSourceTLinks = thrust::raw_pointer_cast(aGraph.mSourceTLinks.data());// n
		//aGraphData.mSinkTLinks = thrust::raw_pointer_cast(aGraph.mSinkTLinks.data());// n
		aGraphData.mTLinks[int(TLinkType::Source)] = thrust::raw_pointer_cast(aGraph.mSourceTLinks.data());// n
		aGraphData.mTLinks[int(TLinkType::Sink)] = thrust::raw_pointer_cast(aGraph.mSinkTLinks.data());// n

		aGraphData.neighbors = thrust::raw_pointer_cast(aGraph.mNeighbors.data());
		aGraphData.secondVertices = thrust::raw_pointer_cast(aGraph.mSecondVertices.data());
		aGraphData.connectionIndices = thrust::raw_pointer_cast(aGraph.mEdges.data());
		aGraphData.mResiduals = thrust::raw_pointer_cast(aGraph.mResiduals.data());
		aGraphData.mSinkFlow = thrust::raw_pointer_cast(aGraph.mSinkFlow.data());
		aGraphData.mVertexCount = aGraph.mLabels.size();
		aGraphData.mEdgeCount = aGraph.mResiduals.size();
}


template<typename TGraphData, typename TPolicy, typename TTraceObject>
struct MinCut
{
	template<PreflowInitialization tPreflowInitialization, bool tDummy>
	struct PreflowInit;

	template<bool tDummy>
	struct PreflowInit<PreflowInitialization::Default, tDummy>
	{
		void
		compute(
			TGraphData &aGraph,
			ParallelQueueView<int> &aVertexQueue,
			std::vector<int> &aLevelStarts)
		{
			CUGIP_DPRINT("push_through_tlinks_from_source");

			dim3 blockSize1D( 512 );
			dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

			pushThroughTLinksFromSourceKernel<<<gridSize1D, blockSize1D>>>(aGraph);

			CUGIP_CHECK_ERROR_STATE("After pushThroughTLinksFromSourceKernel");
			CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		}
	};

	template<bool tDummy>
	struct PreflowInit<PreflowInitialization::Push, tDummy>
	{
		void
		compute(
			TGraphData &aGraph,
			ParallelQueueView<int> &aVertexQueue,
			std::vector<int> &aLevelStarts)
		{
			CUGIP_DPRINT("push_through_tlinks_from_source");

			dim3 blockSize1D( 512 );
			dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

			pushThroughTLinksFromSourceKernel<<<gridSize1D, blockSize1D>>>(aGraph);

			CUGIP_CHECK_ERROR_STATE("After pushThroughTLinksFromSourceKernel");
			CUGIP_CHECK_RESULT(cudaThreadSynchronize());

			typedef RelabelPolicy<512, 64, EdgeForwardTraversable, TLinkType::Source> PreflowRelabelPolicy;
			typedef PushPolicy PreflowPushPolicy;

			Relabel<TGraphData, PreflowRelabelPolicy> relabel;
			Push<TGraphData, PreflowPushPolicy> push;
			relabel.compute(aGraph, aVertexQueue, aLevelStarts, PreflowRelabelPolicy());
			push.compute(aGraph, aVertexQueue, aLevelStarts);
		}
	};


	static float
	compute(
		TGraphData &aGraph,
		ParallelQueueView<int> &aVertexQueue,
		std::vector<int> &aLevelStarts,
		TTraceObject &aTraceObject)
	{
		boost::timer::cpu_timer timer;
		timer.start();
		//CUGIP_DPRINT("MAX FLOW");
		//init_residuals(aGraph);
		aTraceObject.computationStarted(aGraph);
		PreflowInit</*TGraphData, */TPolicy::cPreflowInitialization, true> preflowInit;
		preflowInit.compute(aGraph, aVertexQueue, aLevelStarts);
		//push_through_tlinks_from_source(aGraph);

		//debug_print();
		CUGIP_CHECK_ERROR_STATE("After max_flow init");
		timer.stop();
		//std::cout << timer.format(9, "%w") << "\n";
		bool done = false;
		int iteration = 0;
		//float flow = -1.0f;
		typename TPolicy::RelabelPolicy relabelPolicy;
		//relabelPolicy.edgeTraversalCheck.minimalResidual = 0.7f;
		Relabel<TGraphData, typename TPolicy::RelabelPolicy> relabel;
		Push<TGraphData, typename TPolicy::PushPolicy> push;
		while(!done) {
			timer.start();
			//CUGIP_DPRINT("Relabel");
			aTraceObject.beginIteration(iteration);

			//cudaProfilerStart();
			relabel.compute(aGraph, aVertexQueue, aLevelStarts, relabelPolicy);
			//relabel.compute_dynamic(aGraph, aVertexQueue, aLevelStarts, typename TPolicy::RelabelPolicy());
			//return 0.0f;
			aTraceObject.afterRelabel(iteration, aLevelStarts);

			//cudaProfilerStop();
			//return 0.0f;
			//assign_label_by_distance();
			/*for (int i = max<int>(0, aLevelStarts.size() - 40); i < aLevelStarts.size() - 1; ++i) {
				std::cout << aLevelStarts[i + 1] - aLevelStarts[i] << " ";
			}*/
			//std::copy(begin(aLevelStarts) + , end(aLevelStarts), std::ostream_iterator<int>(std::cout, " "));
			//std::cout << std::endl;
			//break;
			//CUGIP_DPRINT("Push");
			done = !push.compute(aGraph, aVertexQueue, aLevelStarts);
			aTraceObject.afterPush(iteration, done, aGraph);
			//done = !push();
			/*timer.stop();
			CUGIP_TFORMAT(
				"iteration %1%, elapsed time %2% ms, queue size %3%, flow= %4%",
					iteration,
					timer.elapsed().wall / 1000000.0f,
					aVertexQueue.size(),
					computeFlowThroughSinkFrontier(aGraph));*/
			/*float flow2 = computeFlowThroughSinkFrontier(aGraph);
			CUGIP_DPRINT("Flow: " << flow2);
			if (flow == flow2 && !done) {
				throw 3;
			}
			flow = flow2;*/
			//CUGIP_DPRINT("**iteration " << iteration << ": " << timer.format(9, "%w"));
			//CUGIP_DPRINT("Flow: " << computeFlowThroughSinkFrontier(aGraph));
			//if (iteration % 10 == 0)
			//	relabelPolicy.edgeTraversalCheck.minimalResidual = 0.0f;
			++iteration;
		}
		auto flow = computeFlowThroughSinkFrontier(aGraph);
		aTraceObject.computationFinished(flow, aGraph);
		return flow;
	}

	static void
	push_through_tlinks_from_source(TGraphData &aGraph)
	{
		CUGIP_DPRINT("push_through_tlinks_from_source");

		dim3 blockSize1D( 512 );
		dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

		pushThroughTLinksFromSourceKernel<<<gridSize1D, blockSize1D>>>(aGraph);

		CUGIP_CHECK_ERROR_STATE("After pushThroughTLinksFromSourceKernel");
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	static void
	push_through_tlinks_to_sink(TGraphData &aGraph)
	{
		//CUGIP_DPRINT("push_through_tlinks_to_sink");

		dim3 blockSize1D( 512 );
		dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

		pushThroughTLinksToSinkKernel<<<gridSize1D, blockSize1D>>>(aGraph);

		CUGIP_CHECK_ERROR_STATE("After push_through_tlinks_to_sink");
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	static float
	computeFlowThroughSinkFrontier(TGraphData &aGraph)
	{
		push_through_tlinks_to_sink(aGraph);
		return thrust::reduce(
				thrust::device_pointer_cast(aGraph.mSinkFlow),
				thrust::device_pointer_cast(aGraph.mSinkFlow + aGraph.vertexCount()));
	}
};


template<typename TGraph, typename TTraceObject>
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
		graphCutDataFromGraph(aGraph, mGraphData);
		mGraphData.vertexExcess = thrust::raw_pointer_cast(aGraph.mExcess.data()); // n
		mGraphData.labels = thrust::raw_pointer_cast(aGraph.mLabels.data());; // n
		//mGraphData.mSourceTLinks = thrust::raw_pointer_cast(aGraph.mSourceTLinks.data());// n
		//mGraphData.mSinkTLinks = thrust::raw_pointer_cast(aGraph.mSinkTLinks.data());// n
		mGraphData.mTLinks[int(TLinkType::Source)] = thrust::raw_pointer_cast(aGraph.mSourceTLinks.data());// n
		mGraphData.mTLinks[int(TLinkType::Sink)] = thrust::raw_pointer_cast(aGraph.mSinkTLinks.data());// n

		mGraphData.neighbors = thrust::raw_pointer_cast(aGraph.mNeighbors.data());
		mGraphData.secondVertices = thrust::raw_pointer_cast(aGraph.mSecondVertices.data());
		mGraphData.connectionIndices = thrust::raw_pointer_cast(aGraph.mEdges.data());
		mGraphData.mResiduals = thrust::raw_pointer_cast(aGraph.mResiduals.data());
		mGraphData.mSinkFlow = thrust::raw_pointer_cast(aGraph.mSinkFlow.data());
		mGraphData.mVertexCount = aGraph.mLabels.size();
		mGraphData.mEdgeCount = aGraph.mResiduals.size();
		//mGraphData.labels
		//mGraphData.neighbors
		//mGraphData.mSourceTLinks
		//mGraphData.mSinkTLinks
		//mGraphData.secondVertices
		//mGraphData.connectionIndices
		//mGraphData.mResiduals
		//mGraphData.mSinkFlow
		mVertexQueue.reserve(mGraphData.mVertexCount);
	}

	Flow
	run(TTraceObject &aTraceObject)
	{
		CUGIP_ASSERT(mGraphData.connectionIndices != nullptr);
		CUGIP_ASSERT(mGraphData.labels != nullptr);
		CUGIP_ASSERT(mGraphData.mResiduals != nullptr);
		CUGIP_ASSERT(mGraphData.mSinkFlow != nullptr);
		//CUGIP_ASSERT(mGraphData.mSinkTLinks != nullptr);
		//CUGIP_ASSERT(mGraphData.mSourceTLinks != nullptr);
		CUGIP_ASSERT(mGraphData.mTLinks[int(TLinkType::Source)] != nullptr);
		CUGIP_ASSERT(mGraphData.mTLinks[int(TLinkType::Sink)] != nullptr);
		CUGIP_ASSERT(mGraphData.neighbors != nullptr);
		CUGIP_ASSERT(mGraphData.secondVertices != nullptr);
		CUGIP_ASSERT(mGraphData.vertexExcess != nullptr);
		init_residuals();
		return MinCut<GraphCutData<Flow>, GraphCutPolicy, TTraceObject>::compute(
						mGraphData,
						mVertexQueue.view(),
						mLevelStarts,
						aTraceObject);
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

		CUGIP_CHECK_ERROR_STATE("After init_residuals()");
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}


protected:
	TGraph *mGraph;
	GraphCutData<Flow> mGraphData;
	ParallelQueue<int> mVertexQueue;
	std::vector<int> mLevelStarts;
};




} // namespace cugip
