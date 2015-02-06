#pragma once


#include <cugip/math.hpp>
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
#define INVALID_LABEL 1000 //(1 << 30)

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
		EdgeWeight *aWeights);

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
	relabel();

	EdgeList mEdges;
	VertexList mVertices;
	thrust::device_vector<EdgeWeight> mSourceTLinks; // n
	thrust::device_vector<EdgeWeight> mSinkTLinks; // n

	thrust::device_vector<EdgeRecord> mEdgeDefinitions; // m
	thrust::device_vector<EdgeWeight> mEdgeWeights; // m

	thrust::device_vector<EdgeWeight> mExcess; // n
	thrust::device_vector<int> mLabels; // n
	thrust::device_vector<EdgeResidualsRecord> mResiduals; // m

	int mSourceLabel;

	thrust::device_vector<bool> mEnabledVertices; // n
	thrust::device_vector<int> mTemporaryLabels; // n
	thrust::device_vector<EdgeWeight> mSinkFlow; // n

	thrust::device_vector<EdgeWeight> mPushedFlow; // n
};



void
Graph::debug_print()
{
	thrust::host_vector<EdgeWeight> excess = mExcess;
	thrust::host_vector<int> labels = mLabels;

	for (size_t i = 0; i < excess.size(); ++i) {
		if (excess[i] > 0)
			std::cout << i << ": " << excess[i] << " - " << labels[i] << "\n";
	}
}

void
Graph::check_flow()
{
	thrust::host_vector<EdgeResidualsRecord> residuals = mResiduals;
	thrust::host_vector<EdgeWeight> weights = mEdgeWeights;

	for (int i = 0; i < weights.size(); ++i) {
		float diff = 0.0f;
		if (residuals[i].residuals[0] > weights[i]) {
			diff = weights[i] - residuals[i].residuals[1];
		} else {
			diff = weights[i] - residuals[i].residuals[0];
		}
		if (diff < 0.0f) {
			std::cout << "invalid flow - edge: " << i << "\n";
		}
	}
}

void
Graph::set_vertex_count(size_t aCount)
{
	std::cout << cudaMemoryInfoText();
	mSourceTLinks.resize(aCount);
	mSinkTLinks.resize(aCount);
	mExcess.resize(aCount);
	mLabels.resize(aCount);

	mEnabledVertices.resize(aCount);
	mTemporaryLabels.resize(aCount);
	mSinkFlow.resize(aCount);

	mVertices = VertexList(mLabels, mExcess, aCount);

	mSourceLabel = aCount;

	mPushedFlow.resize(aCount);
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
	mResiduals.resize(aEdgeCount);

	thrust::copy(aEdges, aEdges + aEdgeCount, mEdgeDefinitions.begin());
	thrust::copy(aWeights, aWeights + aEdgeCount, mEdgeWeights.begin());

	mEdges = EdgeList(mEdgeDefinitions, mEdgeWeights, mResiduals, aEdgeCount);
}

void
Graph::set_tweights(
	EdgeWeight *aCapSource,
	EdgeWeight *aCapSink)
{
	thrust::copy(aCapSource, aCapSource + mSourceTLinks.size(), mSourceTLinks.begin());
	thrust::copy(aCapSink, aCapSink + mSinkTLinks.size(), mSinkTLinks.begin());

	thrust::fill(mSinkFlow.begin(), mSinkFlow.end(), 0.0f);


	dump_buffer("source_float.raw", aCapSource, mSourceLabel);
	dump_buffer("sink_float.raw", aCapSink, mSourceLabel);
}



void
Graph::init_labels(EdgeList &mEdges, VertexList &mVertices)
{
	thrust::fill(mExcess.begin(), mExcess.end(), 0.0f);
	thrust::fill(mLabels.begin(), mLabels.end(), 0);
}

CUGIP_GLOBAL void
initResidualsKernel(float *aWeights, EdgeResidualsRecord *aResiduals, int aSize)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if (edgeIdx < aSize) {
		aResiduals[edgeIdx].residuals[0] = aWeights[edgeIdx];
		aResiduals[edgeIdx].residuals[1] = aWeights[edgeIdx];
	}
}

void
Graph::init_residuals()
{
	dim3 blockSize1D( 512 );
	dim3 edgeGridSize1D((mEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);

	initResidualsKernel<<<edgeGridSize1D, blockSize1D>>>(
					thrust::raw_pointer_cast(&mEdgeWeights[0]),
					thrust::raw_pointer_cast(&mResiduals[0]),
					mResiduals.size()
					);
}

float
Graph::max_flow()
{
	init_labels(mEdges, mVertices/*, aSourceID, aSinkID */);
	init_residuals();
	//testForCut( aEdges, aVertices, aSourceID, aSinkID );
	push_through_tlinks_from_source();

	assign_label_by_distance();

	thrust::host_vector<int> labels = mLabels;
	dump_buffer("labels_int.raw", &(labels[0]), mSourceLabel);

	bool done = false;
	size_t iteration = 0;
	while(!done) {
		bool push_result = push();
		thrust::host_vector<float> excess = mExcess;
		//dump_buffer(boost::str(boost::format("excess_float_%1%.raw") % iteration), &(excess[0]), mSourceLabel);
		labels = mLabels;
		dump_buffer(boost::str(boost::format("labels_int_%1%.raw") % iteration), &(labels[0]), mSourceLabel);
		thrust::host_vector<float> flow = mPushedFlow;
		dump_buffer(boost::str(boost::format("flow_float_%1%.raw") % iteration), &(flow[0]), mSourceLabel);
		/*debug_print();
		break;*/
		//std::cout << "PUSH\n";
		if (iteration > 2) {
			//debug_print();
			//save_to_graphml("progress.xml");
			break;
		}

		assign_label_by_distance();
		//done = relabel();

		//std::cout << "RELABEL " << iteration << "\n";
		++iteration;

	}
	return thrust::reduce(mSinkFlow.begin(), mSinkFlow.end());
}

CUGIP_GLOBAL void
initBFSKernel(float *aTLinks, int *aLabels, int aSize)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if (vertexIdx < aSize) {
		int label = INVALID_LABEL;
		if (aTLinks[vertexIdx] > 0.0) {
			label = 1;
		}
		aLabels[vertexIdx] = label;
	}
}


CUGIP_GLOBAL void
bfsPropagationKernel(device_flag_view aPropagationSuccessfulFlag, EdgeList aEdges, int *aLabels, int aCurrentLevel)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if (edgeIdx < aEdges.size()) {
		int v1 = aEdges.getEdge(edgeIdx).first;
		int v2 = aEdges.getEdge(edgeIdx).second;
		int l1 = aLabels[v1];
		int l2 = aLabels[v2];
		//TODO - check residuals index
		if (l1 == aCurrentLevel && l2 == INVALID_LABEL && aEdges.getResiduals(edgeIdx).getResidual(v1 > v2) > 0.0f) {
			aLabels[v2] = aCurrentLevel + 1;
			aPropagationSuccessfulFlag.set_device();
		}

		if (l2 == aCurrentLevel && l1 == INVALID_LABEL && aEdges.getResiduals(edgeIdx).getResidual(v2 > v1) > 0.0f) {
			aLabels[v1] = aCurrentLevel + 1;
			aPropagationSuccessfulFlag.set_device();
		}
	}
}


struct compare_value
{
	compare_value(int aVal = 0)
		: val(aVal)
	{}

	__host__ __device__
	bool operator()(int x)
	{
		return x == val;
	}
	int val;
};


void
Graph::assign_label_by_distance()
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (mVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	dim3 edgeGridSize1D( (mEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	initBFSKernel<<<vertexGridSize1D, blockSize1D>>>(
				thrust::raw_pointer_cast(&mSinkTLinks[0]),
				thrust::raw_pointer_cast(&mLabels[0]),
				mVertices.size()
				);

	device_flag propagationSuccessfulFlag;
	size_t currentLevel = 1;
	do {
		propagationSuccessfulFlag.reset_host();
		bfsPropagationKernel<<<edgeGridSize1D, blockSize1D>>>(
				propagationSuccessfulFlag.view(),
				mEdges,
				thrust::raw_pointer_cast(&mLabels[0]),
				currentLevel);
		std::cout << "Level " << currentLevel << "; count = " << thrust::count_if(mLabels.begin(), mLabels.end(), compare_value(currentLevel)) << "\n";

		++currentLevel;
	} while (propagationSuccessfulFlag.check_host());

	std::cout << "BFS levels = " << (currentLevel-1) << "\n";
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
		/*float oldExcess = excess;
		aVertices.getExcess( aVertex ) = excess - pushedFlow;*/
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
	//aVertices.getExcess( aVertex ) += aPushedFlow;
}

CUGIP_DECL_DEVICE bool
tryToPushFromTo(float *aFlow, VertexList &aVertices, int aFrom, int aTo, EdgeList &aEdges, int aEdgeIdx, float residualCapacity )
{
	//printf( "Push successfull\n" );
	if ( residualCapacity > 0 ) {
		float pushedFlow = tryPullFromVertex( aVertices, aFrom, residualCapacity );
		if( pushedFlow > 0.0f ) {
			pushToVertex( aVertices, aTo, pushedFlow );

			atomicAdd( &(aFlow[aTo]), 1/*pushedFlow */);
			updateResiduals( aEdges, aEdgeIdx, pushedFlow, aFrom, aTo );
			return true;
			//aPushSuccessfulFlag.set_device();
			//printf( "Push successfull from %i to %i (edge %i), flow = %f\n", aFrom, aTo, aEdgeIdx, pushedFlow );
		}
	}
	return false;
}

CUGIP_DECL_DEVICE bool
pushThroughEdge(float *aFlow, int edgeIdx, EdgeList &aEdges, VertexList &aVertices)
{
	bool pushed = false;
	int v1, v2;
	int label1, label2;
	EdgeResidualsRecord residualCapacities;
	loadEdgeV1V2L1L2C(aEdges, aVertices, edgeIdx, v1, v2, label1, label2, residualCapacities);
	if (label1 == -1 || label2 == -1) printf("Wrong label");
	if (label1 > label2) {
		pushed = tryToPushFromTo(aFlow, /*aPushSuccessfulFlag, */aVertices, v1, v2, aEdges, edgeIdx, residualCapacities.getResidual(v1 < v2));
	} else if ( label1 < label2 ) {
		pushed = tryToPushFromTo(aFlow, /*aPushSuccessfulFlag, */aVertices, v2, v1, aEdges, edgeIdx, residualCapacities.getResidual(v2 < v1));
	}
	/*if (pushed) {
		printf( "%i -> %i => %i -> %i %f; %f\n", v1, label1, v2, label2, residualCapacities.getResidual( true ), residualCapacities.getResidual( false ) );
	}*/
	return pushed;
}

CUGIP_GLOBAL void
pushKernel(float *aFlow, device_flag_view aPushSuccessfulFlag, EdgeList aEdges, VertexList aVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	//int edgeIdx = blockId * blockDim.x + threadIdx.x;
	int batchIdx = blockId * blockDim.x + threadIdx.x;

	int batchSize = (aEdges.size()/* + 31*/) ;// / 32;

	bool pushed = false;
	if (batchIdx < batchSize) {
		for (int i = 0; i < 1/*32*/; ++i) {
			int edgeIdx = i * batchSize + batchIdx;
			if (edgeIdx < aEdges.size()) {
				pushed = pushThroughEdge(aFlow, edgeIdx, aEdges, aVertices) || pushed;
			}
		}
	}
	pushed = __any(pushed);
	if (threadIdx.x == 0 && pushed) {
		aPushSuccessfulFlag.set_device();
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
pushThroughTLinksToSourceKernel(
		VertexList aVertices,
		float *aTLinks,
		int aSourceLabel
		)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if (vertexIdx < aVertices.size() && aVertices.getLabel(vertexIdx) > aSourceLabel) {
		float excess = aVertices.getExcess(vertexIdx);
		float flow = min(aTLinks[vertexIdx], excess);
		if (flow > 0.0f) {
			aVertices.getExcess(vertexIdx) -= flow;
		}
	}
}


void
Graph::push_through_tlinks_to_source()
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (mVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	pushThroughTLinksToSourceKernel<<< vertexGridSize1D, blockSize1D >>>(
					mVertices,
					thrust::raw_pointer_cast(&mSourceTLinks[0]),
					mSourceLabel
					);
}




CUGIP_GLOBAL void
pushThroughTLinksToSinkKernel(
		VertexList aVertices,
		float *aTLinks,
		float *aSinkFlow
		)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if (vertexIdx < aVertices.size()) {
		float capacity = aTLinks[vertexIdx];
		float excess = aVertices.getExcess(vertexIdx);
		if (capacity > 0.0f && excess > 0.0f) {
			float flow = min(capacity, excess);

			aVertices.getExcess(vertexIdx) -= flow;
			aSinkFlow[vertexIdx] += flow;
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
					thrust::raw_pointer_cast(&mSinkTLinks[0]),
					thrust::raw_pointer_cast(&mSinkFlow[0])
					);
}


bool
Graph::push()
{
	dim3 blockSize1D( 512 );
	int batchSize = (mEdges.size() + 31) / 32;
	dim3 gridSize1D((mEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);
	//dim3 gridSize1D((batchSize + 64*blockSize1D.x - 1) / (64*blockSize1D.x), 64);

	device_flag pushSuccessfulFlag;
	size_t pushIterations = 0;
	do {
		pushSuccessfulFlag.reset_host();
		++pushIterations;
		pushKernel<<<gridSize1D, blockSize1D>>>(thrust::raw_pointer_cast(&(mPushedFlow[0])), pushSuccessfulFlag.view(), mEdges, mVertices);

	} while (pushSuccessfulFlag.check_host());

	push_through_tlinks_to_sink();
	push_through_tlinks_to_source();

	std::cout << "Push iterations = " << pushIterations << "\n";
	return pushIterations == 0;
}


CUGIP_GLOBAL void
getActiveVerticesKernel( VertexList aVertices, bool *aEnabledVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size()) {
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
	//atomicMax(aLabels + aVertexIdx, label);
	atomicMin(aLabels + aVertexIdx, label);
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

		if (v1Enabled && !v2Enabled) { //TODO - check if set to maximum is right
			if (label1 <= label2 && residualCapacities.getResidual(v1 < v2) > 0.0f) {
				//printf("*1 %i-%i; %i - %i %f - current label %i\n", v1, v2, label1, label2, residualCapacities.getResidual(v1 < v2), aLabels[v1]);
				trySetNewHeight(aLabels, v1, label2+1);
			}
			/*if (label1 <= label2 || residualCapacities.getResidual(v1 < v2) <= 0.0f) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v1, label2+1 );
			} else {
				//printf( "%i -> %i, l1 %i l2 %i label1\n", v1, v2, label1, label2 );
				aEnabledVertices[v1] = false;
			}*/
		}
		if (v2Enabled && !v1Enabled) {
			if (label2 <= label1 && residualCapacities.getResidual(v2 < v1) > 0.0f) {
				//printf("*2 %i-%i; %i - %i %f - current label %i\n", v2, v1, label2, label1, residualCapacities.getResidual(v2 < v1), aLabels[v2]);
				trySetNewHeight(aLabels, v2, label1+1);
			}
			/*if( label2 <= label1 || residualCapacities.getResidual(v2 < v1) <= 0.0f) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v2, label1+1 );
			} else {
				aEnabledVertices[v2] = false;
			}*/
		}
	}
}

CUGIP_GLOBAL void
assignNewLabelsKernel(
		VertexList aVertices,
		float *aTLinkWeights,
		bool *aEnabledVertices,
		int *aLabels,
		int aSourceLabel,
		//int aSink,
		device_flag_view aRelabelSuccessfulFlag)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if (vertexIdx < aVertices.size() && aEnabledVertices[vertexIdx]) {
		int oldLabel = aVertices.getLabel( vertexIdx );
		int newLabel = aLabels[vertexIdx];
		if (newLabel > aSourceLabel + 1) {
			if (aTLinkWeights[vertexIdx] > 0.0f) {
				newLabel = aSourceLabel + 1;
			}
		}
		//printf( "vertexIdx %i orig label %i, label = %i\n", vertexIdx, oldLabel, newLabel);
		aVertices.getLabel(vertexIdx) = newLabel;
		aRelabelSuccessfulFlag.set_device();
	}
}


bool
Graph::relabel()
{
	CUGIP_CHECK_ERROR_STATE("Before relabel()");

	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (mVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	dim3 edgeGridSize1D( (mEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	device_flag relabelSuccessfulFlag;
	//int relabelSuccessful;
	//cudaMemcpyToSymbol( "relabelSuccessful", &(relabelSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice );
	//D_COMMAND( M4D::Common::Clock clock; );
	getActiveVerticesKernel<<< vertexGridSize1D, blockSize1D >>>(
					mVertices,
					thrust::raw_pointer_cast(&mEnabledVertices[0])
					);
	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After getActiveVerticesKernel()");

	thrust::fill(mTemporaryLabels.begin(), mTemporaryLabels.end(), MAX_LABEL);
	processEdgesToFindNewLabelsKernel<<<edgeGridSize1D, blockSize1D>>>(
					mEdges,
					mVertices,
					thrust::raw_pointer_cast(&mEnabledVertices[0]),
					thrust::raw_pointer_cast(&mTemporaryLabels[0])//,
					//aSource,
					//aSink
					);
	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After processEdgesToFindNewLabelsKernel()");


	//Sink and source doesn't change height
	//aEnabledVertices[aSource] = false;
	//aEnabledVertices[aSink] = false;
	assignNewLabelsKernel<<<vertexGridSize1D, blockSize1D>>>(
					mVertices,
					thrust::raw_pointer_cast(&mSourceTLinks[0]),
					thrust::raw_pointer_cast(&mEnabledVertices[0]),
					thrust::raw_pointer_cast(&mTemporaryLabels[0]),
					mSourceLabel,
					//aSource,
					//aSink,
					relabelSuccessfulFlag.view()
					);
	cudaThreadSynchronize();
	CUGIP_CHECK_ERROR_STATE("After assignNewLabelsKernel()");


	//cudaMemcpyFromSymbol( &relabelSuccessful, "relabelSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost );
	//D_PRINT( "Relabel result = " << relabelSuccessful << "; took " << clock.SecondsPassed() );
	return !relabelSuccessfulFlag.check_host();
}


}//namespace cugip

#include "graph_to_graphml.hpp"
