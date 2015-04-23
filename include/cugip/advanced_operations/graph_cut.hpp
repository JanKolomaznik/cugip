#pragma once


#include <cugip/math.hpp>
#include <cugip/memory.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/device_flag.hpp>
#include <limits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>


#include <boost/filesystem.hpp>
#include <fstream>

#include <cugip/advanced_operations/detail/graph_cut_implementation.hpp>

namespace cugip {

typedef unsigned NodeId;
typedef unsigned long CombinedNodeId;
struct EdgeRecord
{
	__host__ __device__
	EdgeRecord( NodeId aFirst, NodeId aSecond )
	{
		first = min( aFirst, aSecond );
		second = max( aFirst, aSecond );
	}
	__host__ __device__
	EdgeRecord(): edgeCombIdx(0)
	{ }

	union {
		CombinedNodeId edgeCombIdx;
		struct {
			NodeId second;
			NodeId first;
		};
	};
};


template <typename TFlow>
class Graph
{
public:
	typedef int VertexId;
	typedef TFlow EdgeWeight;

	void
	set_vertex_count(size_t aCount);

	void
	set_nweights(
		size_t aEdgeCount,
		EdgeRecord *aEdges,
		EdgeWeight *aWeightsForward,
		EdgeWeight *aWeightsBackward);

	void
	set_tweights(
		EdgeWeight *aCapSource,
		EdgeWeight *aCapSink);

	TFlow
	max_flow();

	void
	debug_print();
protected:
	void
	init_residuals();

	bool
	bfs_iteration(size_t &aCurrentLevel);
	void
	assign_label_by_distance();

	void
	push_through_tlinks_from_source();

	void
	push_through_tlinks_to_sink();

	bool
	push();

	GraphCutData<TFlow> mGraphData;

	std::vector<int> mLevelStarts;
	ParallelQueue<int> mVertexQueue;
	ParallelQueue<int> mLevelStartsQueue;

	thrust::device_vector<EdgeWeight> mSourceTLinks; // n
	thrust::device_vector<EdgeWeight> mSinkTLinks; // n
	thrust::device_vector<EdgeWeight> mExcess; // n
	thrust::device_vector<int> mLabels; // n
	thrust::device_vector<int> mNeighbors; // n

	thrust::device_vector<int> mSecondVertices; // 2*m
	thrust::device_vector<int> mEdges; // 2*m
	thrust::device_vector<EdgeResidualsRecord<TFlow> > mResiduals; // m

	thrust::device_vector<EdgeWeight> mEdgeWeightsForward; // m
	thrust::device_vector<EdgeWeight> mEdgeWeightsBackward; // m

	thrust::device_vector<EdgeWeight> mSinkFlow; // n
};

/*
template<typename TFlow>
bool
Graph<TFlow>::push()
{
	//CUGIP_DPRINT("push()");
	dim3 blockSize1D(512);
	dim3 gridSize1D((mGraphData.vertexCount() + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);
	device_flag pushSuccessfulFlag;
	size_t pushIterations = 0;
	do {
		pushSuccessfulFlag.reset_host();
		++pushIterations;
		pushKernel<<<gridSize1D, blockSize1D>>>(mGraphData, pushSuccessfulFlag.view());
		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After pushKernel()");
	} while (pushSuccessfulFlag.check_host());
	CUGIP_DPRINT("Push iterations " << pushIterations);
	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After push()");
	//push_through_tlinks_to_sink(); //TODO implement

	//push_through_tlinks_to_source();

	return pushIterations > 1;
}*/

template<typename TFlow>
bool
Graph<TFlow>::push()
{
	thrust::host_vector<int> starts;
	starts.reserve(1000);
	thrust::device_vector<int> device_starts;
	device_starts.reserve(1000);
	//CUGIP_DPRINT("push()");
	dim3 blockSize1D(512);
	device_flag pushSuccessfulFlag;

	for (int i = mLevelStarts.size() - 2; i > 0; --i) {
		int count = mLevelStarts[i] - mLevelStarts[i-1];
		if (count <= blockSize1D.x) {
			starts.push_back(mLevelStarts[i]);
			while (i > 0 && (mLevelStarts[i] - mLevelStarts[i-1]) <= blockSize1D.x) {
				starts.push_back(mLevelStarts[i-1]);
				//CUGIP_DPRINT(i << " - " << mLevelStarts[i-1]);
				--i;
			}
			++i;
			//CUGIP_DPRINT("-------------------------------");
			device_starts = starts;
			dim3 gridSize1D(1);
			pushKernel2<<<gridSize1D, blockSize1D>>>(
					mGraphData,
					mVertexQueue.view(),
					thrust::raw_pointer_cast(device_starts.data()),
					device_starts.size(),
					pushSuccessfulFlag.view());

		} else {
			//CUGIP_DPRINT()
			dim3 gridSize1D((count + blockSize1D.x - 1) / (blockSize1D.x), 1);
			pushKernel<<<gridSize1D, blockSize1D>>>(
					mGraphData,
					mVertexQueue.view(),
					mLevelStarts[i-1],
					mLevelStarts[i],
					pushSuccessfulFlag.view());
		}
	}
	cudaThreadSynchronize();
	//CUGIP_DPRINT("-------------------------------");
	CUGIP_CHECK_ERROR_STATE("After push()");
	return pushSuccessfulFlag.check_host();
}

template<typename TFlow>
bool
Graph<TFlow>::bfs_iteration(size_t &aCurrentLevel)
{
	size_t level = aCurrentLevel;
	dim3 blockSize1D(512, 1, 1);
	int frontierSize = mLevelStarts[aCurrentLevel] - mLevelStarts[aCurrentLevel - 1];
	dim3 levelGridSize1D(1 + (frontierSize - 1) / (blockSize1D.x), 1, 1);
	CUGIP_CHECK_ERROR_STATE("Before bfsPropagationKernel()");
	if (frontierSize <= blockSize1D.x) {
		mLevelStartsQueue.clear();
		bfsPropagationKernel3<<<levelGridSize1D, blockSize1D>>>(
			mVertexQueue.view(),
			mLevelStarts[aCurrentLevel - 1],
			frontierSize,
			mGraphData,
			aCurrentLevel + 1,
			mLevelStartsQueue.view());
		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel3)");
		thrust::host_vector<int> starts;
		mLevelStartsQueue.fill_host(starts);
		int originalStart = mLevelStarts.back();
		int lastStart = originalStart;
		for (int i = 0; i < starts.size(); ++i) {
			if (starts[i] == lastStart) {
				lastStart = -1;
			} else {
				lastStart = starts[i];
			}
			mLevelStarts.push_back(starts[i]);
		}
		aCurrentLevel = mLevelStarts.size() - 1;
		//CUGIP_DPRINT("Level bundle " << (level + 1) << "-" << (aCurrentLevel + 1) << " size: " << (originalStart - mLevelStarts.back()));
		return (lastStart == originalStart) || (lastStart == -1);
	} else {
		bfsPropagationKernel2<<<levelGridSize1D, blockSize1D>>>(
			mVertexQueue.view(),
			mLevelStarts[aCurrentLevel - 1],
			frontierSize,
			mGraphData,
			aCurrentLevel + 1);
		++aCurrentLevel;
		cudaThreadSynchronize();
		CUGIP_CHECK_ERROR_STATE("After bfsPropagationKernel2()");
		int lastLevelSize = mVertexQueue.size();
		//CUGIP_DPRINT("LastLevelSize " << lastLevelSize);
		if (lastLevelSize == mLevelStarts.back()) {
			return true;
		}
		//CUGIP_DPRINT("Level " << (aCurrentLevel + 1) << " size: " << (lastLevelSize - mLevelStarts.back()));
		//if (currentLevel == 2) break;
		mLevelStarts.push_back(lastLevelSize);
	}

	return false;
}

template<typename TFlow>
void
Graph<TFlow>::assign_label_by_distance()
{
	//CUGIP_DPRINT("assign_label_by_distance");
	dim3 blockSize1D(512, 1, 1);
	dim3 gridSize1D((mGraphData.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x), 1);

	mVertexQueue.clear();
	initBFSKernel<<<gridSize1D, blockSize1D>>>(mVertexQueue.view(), mGraphData);

	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After initBFSKernel()");
	int lastLevelSize = mVertexQueue.size();
	//CUGIP_DPRINT("Level 1 size: " << lastLevelSize);
	mLevelStarts.clear();
	mLevelStarts.push_back(0);
	mLevelStarts.push_back(lastLevelSize);
	size_t currentLevel = 1;
	bool finished = lastLevelSize == 0;
	while (!finished) {
		finished = bfs_iteration(currentLevel);
	}

	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After assign_label_by_distance()");
}


template<typename TFlow>
TFlow
Graph<TFlow>::max_flow()
{
	/*testBlockScan<<<1, 512>>>();
	return -1.0f;*/
	//CUGIP_DPRINT("MAX FLOW");
	init_residuals();
	push_through_tlinks_from_source();

	//debug_print();
	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After max_flow init");

	bool done = false;
	size_t iteration = 0;
	while(!done) {
		//CUGIP_DPRINT("**iteration " << iteration);
		assign_label_by_distance();
		//debug_print();
		done = !push();
		++iteration;
	}
	push_through_tlinks_to_sink();
	//CUGIP_DPRINT("Used iterations: " << iteration);
	return thrust::reduce(mSinkFlow.begin(), mSinkFlow.end());
}


template<typename TFlow>
void
Graph<TFlow>::push_through_tlinks_from_source()
{
	//CUGIP_DPRINT("push_through_tlinks_from_source");

	dim3 blockSize1D( 512 );
	dim3 gridSize1D((mGraphData.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

	pushThroughTLinksFromSourceKernel<<<gridSize1D, blockSize1D>>>(mGraphData);

	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After pushThroughTLinksFromSourceKernel");
}

template<typename TFlow>
void
Graph<TFlow>::push_through_tlinks_to_sink()
{
	//CUGIP_DPRINT("push_through_tlinks_to_sink");

	dim3 blockSize1D( 512 );
	dim3 gridSize1D((mGraphData.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

	pushThroughTLinksToSinkKernel<<<gridSize1D, blockSize1D>>>(mGraphData, thrust::raw_pointer_cast(&mSinkFlow[0]));

	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After push_through_tlinks_to_sink");
}


template<typename TFlow>
void
Graph<TFlow>::set_vertex_count(size_t aCount)
{
	mSourceTLinks.resize(aCount);
	mSinkTLinks.resize(aCount);
	mExcess.resize(aCount);
	mLabels.resize(aCount);
	mNeighbors.resize(aCount);
	mSinkFlow.resize(aCount);

	mGraphData.vertexExcess = thrust::raw_pointer_cast(&mExcess[0]); // n
	mGraphData.labels = thrust::raw_pointer_cast(&mLabels[0]);; // n
	mGraphData.mSourceTLinks = thrust::raw_pointer_cast(&mSourceTLinks[0]);// n
	mGraphData.mSinkTLinks = thrust::raw_pointer_cast(&mSinkTLinks[0]);// n
	mGraphData.mVertexCount = aCount;

	mVertexQueue.reserve(aCount);
	mVertexQueue.clear();

	mLevelStartsQueue.reserve(500);
}

template<typename TFlow>
void
Graph<TFlow>::set_nweights(
	size_t aEdgeCount,
	EdgeRecord *aEdges,
	EdgeWeight *aWeightsForward,
	EdgeWeight *aWeightsBackward)
{
	std::vector<std::vector<std::pair<int, int> > > edges(mLabels.size());
	for (size_t i = 0; i < aEdgeCount; ++i) {
		EdgeRecord &edge = aEdges[i];
		edges[edge.first].push_back(std::make_pair(int(i), edge.second));
		edges[edge.second].push_back(std::make_pair(int(i), edge.first));
	}
	thrust::host_vector<int> neighbors(mLabels.size());
	thrust::host_vector<int> secondVertices(2 * aEdgeCount);
	thrust::host_vector<int> edgeIndex(2 * aEdgeCount);

	int start = 0;
	for (size_t i = 0; i < edges.size(); ++i) {
		neighbors[i] = start;
		for (size_t j = 0; j < edges[i].size(); ++j) {
			secondVertices[start + j] = edges[i][j].second;
			edgeIndex[start + j] = edges[i][j].first;
		}
		start += edges[i].size();
	}
	neighbors.push_back(neighbors.back());

	mNeighbors = neighbors;
	mGraphData.neighbors = thrust::raw_pointer_cast(&mNeighbors[0]);
	mSecondVertices = secondVertices;
	mGraphData.secondVertices = thrust::raw_pointer_cast(&mSecondVertices[0]);
	mEdges = edgeIndex;
	mGraphData.connectionIndices = thrust::raw_pointer_cast(&mEdges[0]);

	mEdgeWeightsForward.resize(aEdgeCount);
	mEdgeWeightsBackward.resize(aEdgeCount);
	thrust::copy(aWeightsForward, aWeightsForward + aEdgeCount, mEdgeWeightsForward.begin());
	thrust::copy(aWeightsBackward, aWeightsBackward + aEdgeCount, mEdgeWeightsBackward.begin());

	mResiduals.resize(aEdgeCount);
	mGraphData.mResiduals = thrust::raw_pointer_cast(&mResiduals[0]);
}

template<typename TFlow>
void
Graph<TFlow>::set_tweights(
	EdgeWeight *aCapSource,
	EdgeWeight *aCapSink)
{
	thrust::copy(aCapSource, aCapSource + mSourceTLinks.size(), mSourceTLinks.begin());
	thrust::copy(aCapSink, aCapSink + mSinkTLinks.size(), mSinkTLinks.begin());

	thrust::fill(mSinkFlow.begin(), mSinkFlow.end(), 0.0f);
}

template<typename TFlow>
void
Graph<TFlow>::init_residuals()
{
	//CUGIP_DPRINT("init_residuals()");
	dim3 blockSize1D( 512 );
	dim3 gridSize1D((mResiduals.size() + blockSize1D.x - 1) / (blockSize1D.x), 1);

	initResidualsKernel<<<gridSize1D, blockSize1D>>>(
					thrust::raw_pointer_cast(&mEdgeWeightsForward[0]),
					thrust::raw_pointer_cast(&mEdgeWeightsBackward[0]),
					thrust::raw_pointer_cast(&mResiduals[0]),
					mResiduals.size()
					);

	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After init_residuals()");
}

template<typename TFlow>
void
Graph<TFlow>::debug_print()
{
	thrust::host_vector<EdgeWeight> excess = mExcess;
	thrust::host_vector<int> labels = mLabels;

	for (size_t i = 0; i < excess.size(); ++i) {
		//if (excess[i] > 0)
			std::cout << i << ": " << excess[i] << " - " << labels[i] << "\n";
	}
}

} //namespace cugip

#if 0

template<typename TItem>
void
dump_buffer(const boost::filesystem::path &aPath, const TItem *aBuffer, size_t aCount)
{
	std::ofstream out;
	out.exceptions(std::ofstream::failbit | std::ofstream::badbit);

	out.open(aPath.string().c_str(), std::ios_base::binary);
	out.write(reinterpret_cast<const char *>(aBuffer), aCount * sizeof(TItem));
}


#define MAX_LABEL (1 << 30)
#define INVALID_LABEL (1 << 30)

typedef unsigned NodeId;
typedef unsigned long CombinedNodeId;

struct EdgeResidualsRecord
{
	__host__ __device__
	EdgeResidualsRecord( float aWeight = 0.0f )
	{
		residuals[0] = residuals[1] = aWeight;
	}

	__host__ __device__ float &
	getResidual( bool aFirst )
	{
		return aFirst ? residuals[0] : residuals[1];
	}
	float residuals[2];
};


struct VertexRecord
{
	size_t edgeStart;
};

struct EdgeRecord
{
	__host__ __device__
	EdgeRecord( NodeId aFirst, NodeId aSecond )
	{
		first = min( aFirst, aSecond );
		second = max( aFirst, aSecond );
	}
	__host__ __device__
	EdgeRecord(): edgeCombIdx(0)
	{ }

	union {
		CombinedNodeId edgeCombIdx;
		struct {
			NodeId second;
			NodeId first;
		};
	};
};

struct EdgeList
{
	EdgeList(
		thrust::device_vector< EdgeRecord > &aEdges,
		thrust::device_vector< EdgeResidualsRecord > &aResiduals,
		int aEdgeCount
		)
		: mEdges( aEdges.data().get() )
		, mEdgeResiduals( aResiduals.data().get() )
		, mSize( aEdgeCount )
	{}

	EdgeList()
		: mEdges(NULL)
		, mEdgeResiduals(NULL)
		, mSize(0)
	{}

	__device__ __host__ int
	size()const
	{ return mSize; }

	/*__device__ __host__ int
	residualsCount()const
	{ return 0; }*/

	__device__ EdgeRecord &
	getEdge( int aIdx )
	{
		CUGIP_ASSERT( aIdx < size() );
		CUGIP_ASSERT( mEdges != NULL );
		return mEdges[aIdx];
	}

	/*__device__ int &
	getEdgeIndexToOtherStructures( int aIdx ) const
	{
		CUGIP_ASSERT( aIdx < size() );
		int tmp = mEdgeIndices[ aIdx ];
		CUGIP_ASSERT( tmp < residualsCount() );


		return tmp;
	}*/

	__device__ EdgeResidualsRecord &
	getResiduals( int aIdx )
	{
		CUGIP_ASSERT( aIdx < size() );
		CUGIP_ASSERT( mEdgeResiduals != NULL );
		return mEdgeResiduals[ aIdx ];
	}



	EdgeRecord *mEdges;
	EdgeResidualsRecord *mEdgeResiduals;
	int mSize;
	//int *mEdgeIndices;
};

struct VertexList
{
	VertexList( thrust::device_vector< int >  &aLabels, thrust::device_vector< float >  &aExcess, int aVertexCount )
		: mLabelArray(aLabels.data().get()), mExcessArray(aExcess.data().get()), mSize(aVertexCount)
	{

	}

	VertexList()
		: mLabelArray(NULL), mExcessArray(NULL), mSize(0)
	{}

	__device__ __host__ int
	size()const
	{ return mSize; }

	__device__ float &
	getExcess( int aIdx )
	{
		CUGIP_ASSERT( aIdx < size()/* && aIdx > 0 */);
		return mExcessArray[aIdx];
	}

	__device__ int &
	getLabel( int aIdx )
	{
		CUGIP_ASSERT( aIdx < size()/* && aIdx > 0 */);
		return mLabelArray[aIdx];
	}

	int *mLabelArray;
	float *mExcessArray;
	int mSize;
};

//*********************************************************************************************************


namespace cugip {

class Graph
{
public:
	typedef int VertexId;
	typedef float EdgeWeight;

	void
	set_vertex_count(size_t aCount);

	void
	set_nweights(
		size_t aEdgeCount,
		/*VertexId *aVertices1,
		VertexId *aVertices2,*/
		EdgeRecord *aEdges,
		EdgeWeight *aWeightsForward,
		EdgeWeight *aWeightsBackward);

	void
	set_tweights(
		EdgeWeight *aCapSource,
		EdgeWeight *aCapSink);

	float
	max_flow();

	void
	save_to_graphml(const boost::filesystem::path &file) const;
protected:
	void
	debug_print();

	void
	check_flow();

	void
	init_labels(EdgeList &mEdges, VertexList &mVertices);
	void
	init_residuals();

	void
	push_through_tlinks_from_source();

	void
	push_through_tlinks_to_sink();
	void
	push_through_tlinks_to_source();

	void
	assign_label_by_distance();

	bool
	push();

	bool
	push_meta();

	bool
	push_host();

	bool
	relabel();

	EdgeList mEdges;
	VertexList mVertices;
	thrust::device_vector<EdgeWeight> mSourceTLinks; // n
	thrust::device_vector<EdgeWeight> mSinkTLinks; // n

	thrust::device_vector<EdgeRecord> mEdgeDefinitions; // m
	thrust::device_vector<EdgeWeight> mEdgeWeightsForward; // m
	thrust::device_vector<EdgeWeight> mEdgeWeightsBackward; // m

	thrust::device_vector<EdgeWeight> mExcess; // n
	thrust::device_vector<int> mLabels; // n
	thrust::device_vector<EdgeResidualsRecord> mResiduals; // m

	int mSourceLabel;

	thrust::device_vector<bool> mEnabledVertices; // n
	thrust::device_vector<int> mTemporaryLabels; // n
	thrust::device_vector<EdgeWeight> mSinkFlow; // n

	thrust::device_vector<EdgeWeight> mPushedFlow; // n
};



#include "graph_cut.tcc"

}//namespace cugip

#include "graph_to_graphml.hpp"

#endif

