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
protected:
	void
	assign_label_by_distance();

	void
	push_through_tlinks_from_source();

	bool
	push();

	GraphCutData<TFlow> mGraphData;

	ParallelQueue<int> mVertexQueue;

	thrust::device_vector<EdgeWeight> mSourceTLinks; // n
	thrust::device_vector<EdgeWeight> mSinkTLinks; // n
	thrust::device_vector<EdgeWeight> mExcess; // n
	thrust::device_vector<int> mLabels; // n

	thrust::device_vector<EdgeResidualsRecord<TFlow> > mResiduals; // m

	//thrust::device_vector<int> mSecondVerte
/*
	EdgeList mEdges;
	VertexList mVertices;

	thrust::device_vector<EdgeRecord> mEdgeDefinitions; // m
	thrust::device_vector<EdgeWeight> mEdgeWeightsForward; // m
	thrust::device_vector<EdgeWeight> mEdgeWeightsBackward; // m


	int mSourceLabel;

	thrust::device_vector<bool> mEnabledVertices; // n
	thrust::device_vector<int> mTemporaryLabels; // n
	thrust::device_vector<EdgeWeight> mSinkFlow; // n

	thrust::device_vector<EdgeWeight> mPushedFlow; // n
*/
};


template<typename TFlow>
bool
Graph<TFlow>::push()
{
	dim3 blockSize1D(512);
	dim3 gridSize1D((mGraphData.vertexCount() + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);
	device_flag pushSuccessfulFlag;
	size_t pushIterations = 0;
	do {
		pushSuccessfulFlag.reset_host();
		++pushIterations;
		pushKernel<<<gridSize1D, blockSize1D>>>(mGraphData, pushSuccessfulFlag.view());
	} while (pushSuccessfulFlag.check_host());

	//push_through_tlinks_to_sink(); //TODO implement

	//push_through_tlinks_to_source();

	return pushIterations > 1;
}


template<typename TFlow>
void
Graph<TFlow>::assign_label_by_distance()
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D((mGraphData.vertexCount() + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);

	/*initBFSKernel<<<vertexGridSize1D, blockSize1D>>>(
				thrust::raw_pointer_cast(&mSinkTLinks[0]),
				thrust::raw_pointer_cast(&mLabels[0]),
				mVertices.size()
				);*/

	std::vector<int> levelStarts;
	device_flag propagationSuccessfulFlag;
	size_t currentLevel = 1;
	do {
		propagationSuccessfulFlag.reset_host();
		bfsPropagationKernel<<<gridSize1D, blockSize1D>>>(
				mVertexQueue,
				levelStarts[currentLevel - 1],
				levelStarts[currentLevel] - levelStarts[currentLevel - 1],
				mGraphData,
				currentLevel,
				propagationSuccessfulFlag.view());
		levelStarts.push_back(mVertexQueue.size());
		++currentLevel;
	} while (propagationSuccessfulFlag.check_host());
}


template<typename TFlow>
TFlow
Graph<TFlow>::max_flow()
{
	//init_residuals();
	push_through_tlinks_from_source();

	bool done = false;
	size_t iteration = 0;
	while(!done) {
		assign_label_by_distance();
		done = !push();
		++iteration;
	}
	return 0;//thrust::reduce(mSinkFlow.begin(), mSinkFlow.end());
}


template<typename TFlow>
void
Graph<TFlow>::push_through_tlinks_from_source()
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D((mGraphData.vertexCount() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	pushThroughTLinksFromSourceKernel<<<gridSize1D, blockSize1D>>>(mGraphData);
}


void
Graph::set_vertex_count(size_t aCount)
{
	mSourceTLinks.resize(aCount);
	mSinkTLinks.resize(aCount);
	mExcess.resize(aCount);
	mLabels.resize(aCount);

	/*mEnabledVertices.resize(aCount);
	mTemporaryLabels.resize(aCount);
	mSinkFlow.resize(aCount);

	mVertices = VertexList(mLabels, mExcess, aCount);

	mSourceLabel = aCount;

	mPushedFlow.resize(aCount);*/
}

void
Graph::set_nweights(
	size_t aEdgeCount,
	EdgeRecord *aEdges,
	EdgeWeight *aWeightsForward,
	EdgeWeight *aWeightsBackward)
{
	mEdgeDefinitions.resize(aEdgeCount);
	mEdgeWeightsForward.resize(aEdgeCount);
	mEdgeWeightsBackward.resize(aEdgeCount);
	mResiduals.resize(aEdgeCount);

	thrust::copy(aEdges, aEdges + aEdgeCount, mEdgeDefinitions.begin());
	thrust::copy(aWeightsForward, aWeightsForward + aEdgeCount, mEdgeWeightsForward.begin());
	thrust::copy(aWeightsBackward, aWeightsBackward + aEdgeCount, mEdgeWeightsBackward.begin());

	mEdges = EdgeList(mEdgeDefinitions, mResiduals, aEdgeCount);
}

void
Graph::set_tweights(
	EdgeWeight *aCapSource,
	EdgeWeight *aCapSink)
{
	thrust::copy(aCapSource, aCapSource + mSourceTLinks.size(), mSourceTLinks.begin());
	thrust::copy(aCapSink, aCapSink + mSinkTLinks.size(), mSinkTLinks.begin());

	thrust::fill(mSinkFlow.begin(), mSinkFlow.end(), 0.0f);
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

