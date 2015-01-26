#pragma once


#include <cugip/math.hpp>
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/device_flag.hpp>
#include <limits>

#include <thrust/device_vector.h>

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
		thrust::device_vector< float > &aWeights,
		thrust::device_vector< EdgeResidualsRecord > &aResiduals,
		int aEdgeCount
		)
		: mEdges( aEdges.data().get() )
		, mWeights( aWeights.data().get() )
		, mEdgeResiduals( aResiduals.data().get() )
		, mSize( aEdgeCount )
	{}

	EdgeList()
		: mEdges(NULL)
		, mWeights(NULL)
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
	float *mWeights;
	EdgeResidualsRecord *mEdgeResiduals;
	int mSize;
	//int *mEdgeIndices;
};

struct VertexList
{
	VertexList( thrust::device_vector< int >  &aLabels, thrust::device_vector< float >  &aExcess, int aVertexCount )
		: mLabelArray( aLabels.data().get() ), mExcessArray( aExcess.data().get() ), mSize( aVertexCount + 1 )
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
		CUGIP_ASSERT( aIdx < size() && aIdx > 0 );
		return mExcessArray[aIdx];
	}

	__device__ int &
	getLabel( int aIdx )
	{
		CUGIP_ASSERT( aIdx < size() && aIdx > 0 );
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
		EdgeWeight *aWeights);

	void
	set_tweights(
		EdgeWeight *aCapSource,
		EdgeWeight *aCapSink);

	void
	max_flow();
protected:
	void
	init_labels(EdgeList &mEdges, VertexList &mVertices);
	void
	init_residuals();

	void
	push_through_tlinks_from_source();

	void
	push_through_tlinks_to_sink();

	bool
	push();

	bool
	relabel();

	EdgeList mEdges;
	VertexList mVertices;
	thrust::device_vector<EdgeWeight> mSourceTLinks; // n
	thrust::device_vector<EdgeWeight> mSinkTLinks; // n

	thrust::device_vector<EdgeRecord> mEdgeDefinitions; // m
	thrust::device_vector<EdgeWeight> mEdgeWeights; // m

	thrust::device_vector<EdgeWeight> mExcess; // n
	thrust::device_vector<int> mLabels; // n
	thrust::device_vector< EdgeResidualsRecord > mResiduals; // m

	int mSourceLabel;

	thrust::device_vector<bool> mEnabledVertices;
};


void
Graph::set_vertex_count(size_t aCount)
{
	mSourceTLinks.resize(aCount);
	mSinkTLinks.resize(aCount);
	mExcess.resize(aCount);
	mLabels.resize(aCount);

	mVertices = VertexList(mLabels, mExcess, aCount);
}

void
Graph::set_nweights(
	size_t aEdgeCount,
	/*VertexId *aVertices1,
	VertexId *aVertices2,*/
	EdgeRecord *aEdges,
	EdgeWeight *aWeights)
{
	mEdgeDefinitions.resize(aEdgeCount);
	mEdgeWeights.resize(aEdgeCount);

	thrust::copy(aEdges, aEdges + aEdgeCount, mEdgeDefinitions.begin());
	thrust::copy(aEdgeWeights, aEdgeWeights + aEdgeCount, mEdgeWeights.begin());

	mEdges = EdgeList(mEdgeDefinitions, mEdgeWeights, mResiduals, aEdgeCount);
}

void
Graph::set_tweights(
	EdgeWeight *aCapSource,
	EdgeWeight *aCapSink)
{
	thrust::copy(aCapSource, aCapSource + mSourceTLinks.size(), mSourceTLinks.begin());
	thrust::copy(aCapSink, aCapSink + mSinkTLinks.size(), mSinkTLinks.begin());
}



void
Graph::init_labels(EdgeList &mEdges, VertexList &mVertices)
{
	thrust::fill(mExcess.begin(), mExcess.end(), 0.0f);
	thrust::fill(mLabels.begin(), mLabels.end(), 0);
}


void
Graph::init_residuals()
{
//	thrust::fill(mResiduals.begin(), mResiduals.end(), 0.0f);
}

void
Graph::max_flow()
{
	init_labels(mEdges, mVertices/*, aSourceID, aSinkID */);
	//testForCut( aEdges, aVertices, aSourceID, aSinkID );
	push_through_tlinks_from_source();

	bool done = false;
	size_t iteration = 0;
	while(!done) {
		bool push_result = push();// aEdges, aVertices, aSourceID, aSinkID );

		done = relabel();// aEdges, aVertices, tmpEnabledVertex, tmpLabels, aSourceID, aSinkID );
		++iteration;

		//D_PRINT( "Finished iteration n.: " << iteration << "; Push sucessful = " << pushRes << "; seconds passed: " << clock.secondsPassed() );
		/*if( iteration % 20 == 0 ) {
			//globalRelabel( aEdges, aVertices, aSourceID, aSinkID );
			//testForCut( aEdges, aVertices, aSourceID, aSinkID );
		}*/
	}
}

//*********************************************************************************************************


inline CUGIP_DECL_DEVICE void
loadEdgeV1V2L1L2C( EdgeList &aEdges, VertexList &aVertices, int aEdgeIdx, int &aV1, int &aV2, int &aLabel1, int &aLabel2, EdgeResidualsRecord &aResidualCapacities )
{
	EdgeRecord rec = aEdges.getEdge( aEdgeIdx );
	aV1 = rec.first;
	aV2 = rec.second;

	aLabel1 = aVertices.getLabel( aV1 );
	aLabel2 = aVertices.getLabel( aV2 );

	aResidualCapacities = aEdges.getResiduals( aEdgeIdx );
}

inline CUGIP_DECL_DEVICE void
updateResiduals( EdgeList &aEdges, int aEdgeIdx, float aPushedFlow, int aFrom, int aTo )
{
	EdgeResidualsRecord &residuals = aEdges.getResiduals( aEdgeIdx );

	residuals.getResidual( aFrom < aTo ) -= aPushedFlow;
	residuals.getResidual( !(aFrom < aTo) ) += aPushedFlow;
}

inline CUGIP_DECL_DEVICE float
tryPullFromVertex( VertexList &aVertices, int aVertex, float aResidualCapacity )
{
	float excess = aVertices.getExcess(aVertex);
	float pushedFlow;
	while ( excess > 0.0f ) {
		pushedFlow = min( excess, aResidualCapacity );
		float oldExcess = atomicFloatCAS(&(aVertices.getExcess( aVertex )), excess, excess - pushedFlow);
		if(excess == oldExcess) {
			return pushedFlow;
		} else {
			excess = oldExcess;
		}
	}
	return 0.0f;
}

inline CUGIP_DECL_DEVICE void
pushToVertex( VertexList &aVertices, int aVertex, float aPushedFlow )
{
	atomicAdd( &(aVertices.getExcess( aVertex )), aPushedFlow );
}

CUGIP_DECL_DEVICE void
tryToPushFromTo(device_flag_view &aPushSuccessfulFlag, VertexList &aVertices, int aFrom, int aTo, EdgeList &aEdges, int aEdgeIdx, float residualCapacity )
{
	//printf( "Push successfull\n" );
	if ( residualCapacity > 0 ) {
		float pushedFlow = tryPullFromVertex( aVertices, aFrom, residualCapacity );
		if( pushedFlow > 0.0f ) {
			pushToVertex( aVertices, aTo, pushedFlow );
			updateResiduals( aEdges, aEdgeIdx, pushedFlow, aFrom, aTo );
			aPushSuccessfulFlag.set_device();
			//printf( "Push successfull from %i to %i (edge %i), flow = %f\n", aFrom, aTo, aEdgeIdx, pushedFlow );
		}
	}
}


CUGIP_GLOBAL void
pushKernel(device_flag_view aPushSuccessfulFlag, EdgeList aEdges, VertexList aVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if (edgeIdx < aEdges.size()) {
		int v1, v2;
		int label1, label2;
		EdgeResidualsRecord residualCapacities;
		loadEdgeV1V2L1L2C(aEdges, aVertices, edgeIdx, v1, v2, label1, label2, residualCapacities);
		//printf( "%i -> %i => %i -> %i %f; %f\n", v1, label1, v2, label2, residualCapacities.getResidual( true ), residualCapacities.getResidual( false ) );
		if (label1 > label2) {
			tryToPushFromTo(aPushSuccessfulFlag, aVertices, v1, v2, aEdges, edgeIdx, residualCapacities.getResidual(v1 < v2));
		} else if ( label1 < label2 ) {
			tryToPushFromTo(aPushSuccessfulFlag, aVertices, v2, v1, aEdges, edgeIdx, residualCapacities.getResidual(v2 < v1));
		}
	}
}

//*********************************************************************************************************

CUGIP_GLOBAL void
pushThroughTLinksFromSourceKernel(
		VertexList aVertices,
		float *aTLinks
		)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if (vertexIdx < aVertices.size()) {
		float capacity = aTLinks[vertexIdx];
		if (capacity > 0.0) {
			aVertices.getExcess(vertexIdx) += capacity;
		}
	}
}


void
Graph::push_through_tlinks_from_source()
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (mVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	pushThroughTLinksFromSourceKernel<<< vertexGridSize1D, blockSize1D >>>(
					mVertices,
					thrust::raw_pointer_cast(&mSourceTLinks[0])
					);
}

CUGIP_GLOBAL void
pushThroughTLinksToSinkKernel(
		VertexList aVertices,
		float *aTLinks
		)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if (vertexIdx < aVertices.size()) {
		float capacity = aTLinks[vertexIdx];
		if (capacity > 0.0) {
			aVertices.getExcess(vertexIdx) += capacity;
		}
	}
}


void
Graph::push_through_tlinks_to_sink()
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (mVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	pushThroughTLinksToSinkKernel<<< vertexGridSize1D, blockSize1D >>>(
					mVertices,
					thrust::raw_pointer_cast(&mSinkTLinks[0])
					);
}


bool
Graph::push()
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D((mEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);

	//thrust::device_ptr<float > sourceExcess( mVertices.mExcessArray + aSource );
	//thrust::device_ptr<float > sinkExcess( mVertices.mExcessArray + aSink );

	device_flag pushSuccessfulFlag;
	//D_PRINT( "gridSize1D " << gridSize1D.x << "; " << gridSize1D.y << "; " << gridSize1D.z );
	//static const float SOURCE_EXCESS = 1000000.0f;
	//D_COMMAND( M4D::Common::Clock clock; );
	size_t pushIterations = 0;
	do {
		pushSuccessfulFlag.reset_host();
		//CUDA_CHECK_RESULT( cudaMemcpy( (void*)(aVertices.mExcessArray + aSource), (void*)(&SOURCE_EXCESS), sizeof(float), cudaMemcpyHostToDevice ) );
	//	*sourceExcess = SOURCE_EXCESS;

		++pushIterations;
	//	CUDA_CHECK_RESULT( cudaMemcpyToSymbol( "pushSuccessful", &(pushSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice ) );

		//CheckCudaErrorState( TO_STRING( "Before push kernel n. " << pushIterations ) );
		pushKernel<<<gridSize1D, blockSize1D>>>(pushSuccessfulFlag.view(), mEdges, mVertices);
		//CheckCudaErrorState( TO_STRING( "After push iteration n. " << pushIterations ) );

	//	CUDA_CHECK_RESULT( cudaThreadSynchronize() );

	//	CUDA_CHECK_RESULT( cudaMemcpyFromSymbol( &pushSuccessful, "pushSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost ) );

		//D_PRINT( "-----------------------------------" );

	} while (pushSuccessfulFlag.check_host());

	push_through_tlinks_to_sink();
	//D_PRINT( "Push iteration count = " << pushIterations << "; Sink excess = " << *sinkExcess << "; took " << clock.SecondsPassed());
	return pushIterations == 0;
}


CUGIP_GLOBAL void
getVerticesWithExcessKernel( VertexList aVertices, bool *aEnabledVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx > 0 ) {
		aEnabledVertices[vertexIdx] = aVertices.getExcess(vertexIdx) > 0.0f;
	}
}

/*__device__ void
loadEdgeV1V2L1L2( EdgeList &aEdges, VertexList &aVertices, int aEdgeIdx, int &aV1, int &aV2, int &aLabel1, int &aLabel2 )
{
	EdgeRecord rec = aEdges.getEdge( aEdgeIdx );
	aV1 = rec.first;
	aV2 = rec.first;

	aLabel1 = aVertices.getLabel( aV1 );
	aLabel2 = aVertices.getLabel( aV2 );
}*/

CUGIP_DECL_DEVICE void
trySetNewHeight( int *aLabels, int aVertexIdx, int label )
{
	atomicMax(aLabels + aVertexIdx, label);
}

CUGIP_GLOBAL void
processEdgesToFindNewLabelsKernel(
		EdgeList aEdges,
		VertexList aVertices,
		bool *aEnabledVertices,
		int *aLabels/*,
		int aSource, int aSink*/)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if (edgeIdx < aEdges.size()) {
		int v1, v2;
		int label1, label2;
		EdgeResidualsRecord residualCapacities;
		loadEdgeV1V2L1L2C( aEdges, aVertices, edgeIdx, v1, v2, label1, label2, residualCapacities );

		bool v1Enabled = aEnabledVertices[v1];
		bool v2Enabled = aEnabledVertices[v2];

		if (v1Enabled) { //TODO - check if set to maximum is right
			if (label1 <= label2 || residualCapacities.getResidual(v1 < v2) <= 0.0f/* || v2 == aSink*//* || v2 == aSource*/ ) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v1, label2+1 );
			} else {
				//printf( "%i -> %i, l1 %i l2 %i label1\n", v1, v2, label1, label2 );
				aEnabledVertices[v1] = false;
			}
		}
		if (v2Enabled) {
			if( label2 <= label1 || residualCapacities.getResidual(v2 < v1) <= 0.0f/* || v1 == aSink*//* || v1 == aSource*/  ) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v2, label1+1 );
			} else {
				aEnabledVertices[v2] = false;
			}
		}
	}
}

CUGIP_GLOBAL void
assignNewLabelsKernel(
		VertexList aVertices,
		bool *aEnabledVertices,
		int *aLabels,
		//int aSource,
		//int aSink,
		device_flag_view aRelabelSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx >= 0 ) {
		if ( /*vertexIdx != aSink && */aEnabledVertices[ vertexIdx ] ) {
			//printf( "vertexIdx %i orig label %i, label = %i excess = %f\n", vertexIdx, aVertices.getLabel( vertexIdx ), aLabels[ vertexIdx ], aVertices.mExcessArray[ vertexIdx ] );
			int newLabel = aLabels[ vertexIdx ];
			//if (newLabel != MAX_LABEL)
			{
				aVertices.getLabel( vertexIdx ) = newLabel;
				//if ( vertexIdx != aSource ) {
					aRelabelSuccessfulFlag.set_device();
					//relabelSuccessful = 1;
					//printf("relabel %i\n", vertexIdx);
				//}
			}
		}
	}
}


bool
Graph::relabel()
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (mVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	dim3 edgeGridSize1D( (mEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	device_flag relabelSuccessfulFlag;
	//int relabelSuccessful;
	//cudaMemcpyToSymbol( "relabelSuccessful", &(relabelSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice );
	//D_COMMAND( M4D::Common::Clock clock; );
	getVerticesWithExcessKernel<<< vertexGridSize1D, blockSize1D >>>(
					mVertices,
					thrust::raw_pointer_cast(&mEnabledVertices[0])
					);
	//cudaThreadSynchronize();
	//CheckCudaErrorState( "After getVerticesWithExcessKernel()" );

	//thrust::fill( aLabels.begin(), aLabels.end(), MAX_LABEL );
	processEdgesToFindNewLabelsKernel<<<edgeGridSize1D, blockSize1D>>>(
					mEdges,
					mVertices,
					thrust::raw_pointer_cast(&mEnabledVertices[0]),
					thrust::raw_pointer_cast(&mLabels[0])//,
					//aSource,
					//aSink
					);
	//cudaThreadSynchronize();
	//CheckCudaErrorState( "After processEdgesToFindNewLabelsKernel()" );


	//Sink and source doesn't change height
	//aEnabledVertices[aSource] = false;
	//aEnabledVertices[aSink] = false;
	assignNewLabelsKernel<<< vertexGridSize1D, blockSize1D >>>(
					mVertices,
					thrust::raw_pointer_cast(&mEnabledVertices[0]),
					thrust::raw_pointer_cast(&mLabels[0]),
					//aSource,
					//aSink,
					relabelSuccessfulFlag.view()
					);
	//cudaThreadSynchronize();
	//CheckCudaErrorState( "After assignNewLabelsKernel()" );


	//cudaMemcpyFromSymbol( &relabelSuccessful, "relabelSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost );
	//D_PRINT( "Relabel result = " << relabelSuccessful << "; took " << clock.SecondsPassed() );
	return !relabelSuccessfulFlag.check_host();
}


}//namespace cugip
