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

#define INVALID_LABEL (1 << 30)

namespace cugip {

template<typename TFlow>
struct EdgeResidualsRecord
{
	__host__ __device__
	EdgeResidualsRecord( TFlow aWeight = 0.0f )
	{
		residuals[0] = residuals[1] = aWeight;
	}

	__host__ __device__ float &
	getResidual( bool aFirst )
	{
		return aFirst ? residuals[0] : residuals[1];
	}
	TFlow residuals[2];
};


template<typename TType>
class ParallelQueue
{
public:
	CUGIP_DECL_DEVICE int
	allocate(int aItemCount)
	{
		return atomicAdd(mSize.get(), aItemCount);
	}

	CUGIP_DECL_DEVICE int
	append(const TType &aItem)
	{
		int index = atomicAdd(mSize.get(), 1);
		mData[index] = aItem;
		return index;
	}

	CUGIP_DECL_HOST int
	size()
	{
		return mSize.retrieve_host();
	}

	CUGIP_DECL_DEVICE TType &
	get_device(int aIndex)
	{
		return mData[aIndex];
	}

	TType *mData;
	device_ptr<int> mSize;
};

template<typename TFlow>
struct GraphCutData
{
	CUGIP_DECL_DEVICE int
	neighborCount(int aVertexId)
	{
		return firstNeighborIndex(aVertexId + 1) - firstNeighborIndex(aVertexId);
	}

	CUGIP_DECL_DEVICE TFlow &
	excess(int aVertexId)
	{
		return vertexExcess[aVertexId];
	}

	CUGIP_DECL_DEVICE int &
	label(int aVertexId)
	{
		return labels[aVertexId];
	}

	CUGIP_DECL_DEVICE int
	vertexCount()
	{
		return mVertexCount;
	}

	CUGIP_DECL_DEVICE int
	firstNeighborIndex(int aVertexId)
	{
		return neighbors[aVertexId];
	}

	CUGIP_DECL_DEVICE int
	secondVertex(int aIndex)
	{
		return edgeVertices[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow
	sourceTLinkCapacity(int aIndex)
	{
		return mSourceTLinks[aIndex];
	}

	CUGIP_DECL_DEVICE EdgeResidualsRecord<TFlow> &
	residuals(int aIndex)
	{
		return mResiduals[aIndex];
	}


	int mVertexCount;

	TFlow *vertexExcess; // n
	int *labels; // n
	int *neighbors; // n

	int *edgeVertices; // m

	TFlow *mSourceTLinks; // n
	TFlow *mSinkTLinks; // n

	EdgeResidualsRecord<TFlow> *mResiduals; // m

};

template<typename TFlow>
CUGIP_GLOBAL void
bfsPropagationKernel(
		ParallelQueue<int> aVertices,
		int aStart,
		int aCount,
		GraphCutData<TFlow> aGraph,
		int aCurrentLevel,
		device_flag_view aPropagationSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aCount) {
		int vertex = aVertices.get_device(aStart + index);
		int neighborCount = aGraph.neighborCount(vertex);
		int firstNeighborIndex = aGraph.firstNeighborIndex(vertex);
		for (int i = 0; i < neighborCount; ++i) {
			int secondVertex = aGraph.secondVertex(firstNeighborIndex + i);
			int label = aGraph.label(secondVertex);
			if (label == INVALID_LABEL) {
				aGraph.label(secondVertex) = aCurrentLevel;
				aVertices.append(secondVertex);
				aPropagationSuccessfulFlag.set_device();
			}
		}
	}
}

template<typename TFlow>
CUGIP_DECL_DEVICE TFlow
tryPull(GraphCutData<TFlow> &aGraph, int aFrom, int aTo, TFlow aCapacity)
{
	TFlow excess = aGraph.excess(aFrom);
	while (excess > 0.0f) {
		float pushedFlow = min(excess, aCapacity);
		TFlow oldExcess = atomicFloatCAS(&(aGraph.excess(aFrom)), excess, excess - pushedFlow);
		if(excess == oldExcess) {
			return pushedFlow;
		} else {
			excess = oldExcess;
		}
	}

	return 0;
}


template<typename TFlow>
CUGIP_DECL_DEVICE bool
tryPullPush(GraphCutData<TFlow> &aGraph, int aFrom, int aTo, int aConnectionIndex)
{
	EdgeResidualsRecord<TFlow> &edge = aGraph.residuals(aConnectionIndex);
	TFlow residual = edge.getResidual(aFrom < aTo);
	TFlow flow = tryPull(aGraph, aFrom, aTo, residual);
	if (flow > 0) {
		atomicAdd(&(aGraph.excess(aFrom)), flow);
		edge.getResidual(aFrom < aTo) -= flow;
		edge.getResidual(aFrom > aTo) += flow;
		return true;
	}
	return false;
}

template<typename TFlow>
CUGIP_GLOBAL void
pushKernel(
		GraphCutData<TFlow> aGraph,
		device_flag_view aPushSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		int neighborCount = aGraph.neighborCount(index);
		int firstNeighborIndex = aGraph.firstNeighborIndex(index);
		int label = aGraph.label(index);
		for (int i = 0; i < neighborCount; ++i) {
			int secondVertex = aGraph.secondVertex(firstNeighborIndex + i);
			int secondLabel = aGraph.label(secondVertex);
			if (label > secondLabel) {
				if (tryPullPush(aGraph, index, secondVertex, firstNeighborIndex + i)) {
					aPushSuccessfulFlag.set_device();
				}
			}
		}
	}
}


template<typename TFlow>
CUGIP_GLOBAL void
pushThroughTLinksFromSourceKernel(GraphCutData<TFlow> aGraph)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	if (index < aGraph.vertexCount()) {
		float capacity = aGraph.sourceTLinkCapacity(index);
		if (capacity > 0.0) {
			aGraph.excess(index) += capacity;
		}
	}
}



} // namespace cugip
/*


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

	//__device__ __host__ int
	//residualsCount()const
	//{ return 0; }

	__device__ EdgeRecord &
	getEdge( int aIdx )
	{
		CUGIP_ASSERT( aIdx < size() );
		CUGIP_ASSERT( mEdges != NULL );
		return mEdges[aIdx];
	}

	//__device__ int &
	//getEdgeIndexToOtherStructures( int aIdx ) const
	//{
	//	CUGIP_ASSERT( aIdx < size() );
	//	int tmp = mEdgeIndices[ aIdx ];
	//	CUGIP_ASSERT( tmp < residualsCount() );


	//	return tmp;
	//}

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
		CUGIP_ASSERT( aIdx < size());
		return mExcessArray[aIdx];
	}

	__device__ int &
	getLabel( int aIdx )
	{
		CUGIP_ASSERT( aIdx < size());
		return mLabelArray[aIdx];
	}

	int *mLabelArray;
	float *mExcessArray;
	int mSize;
};
*/
//*********************************************************************************************************


namespace cugip {
/*
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
		//VertexId *aVertices1,
		//VertexId *aVertices2,
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

*/


}//namespace cugip

