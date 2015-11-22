#pragma once


namespace cugip {

enum {
	CONNECTION_VERTEX = 1 << 31,
	CONNECTION_INDEX_MASK = ~CONNECTION_VERTEX
};

template<typename TFlow>
struct EdgeResidualsRecord
{
	CUGIP_DECL_HYBRID
	EdgeResidualsRecord( TFlow aWeight = 0.0f )
	{
		residuals[0] = residuals[1] = aWeight;
	}

	CUGIP_DECL_HYBRID float &
	getResidual( bool aFirst )
	{
		return aFirst ? residuals[0] : residuals[1];
	}
	TFlow residuals[2];
};

template<typename TFlow>
struct GraphCutData
{
	typedef TFlow Flow;
	CUGIP_DECL_DEVICE int
	neighborCount(int aVertexId) const
	{
		//if (aVertexId < 0) printf("neighborCount()\n");
		return firstNeighborIndex(aVertexId + 1) - firstNeighborIndex(aVertexId);
	}

	CUGIP_DECL_DEVICE TFlow &
	excess(int aVertexId)
	{
		//if (aVertexId < 0) printf("excess()\n");
		return vertexExcess[aVertexId];
	}

	CUGIP_DECL_DEVICE int &
	label(int aVertexId)
	{
		//if (aVertexId < 0) printf("label()\n");
		return labels[aVertexId];
	}

	CUGIP_DECL_HYBRID int
	vertexCount() const
	{
		return mVertexCount;
	}

	CUGIP_DECL_DEVICE int
	firstNeighborIndex(int aVertexId) const
	{
		//if (aVertexId < 0) printf("firstNeighborIndex()\n");
		return neighbors[aVertexId];
	}

	CUGIP_DECL_DEVICE int
	secondVertex(int aIndex) const
	{
		//if (aIndex < 0) printf("secondVertex()\n");
		return secondVertices[aIndex];
	}

	CUGIP_DECL_DEVICE int
	connectionIndex(int aIndex) const
	{
		//if (aIndex < 0) printf("connectionIndex()\n");
		return CONNECTION_INDEX_MASK & connectionIndices[aIndex];
	}

	CUGIP_DECL_DEVICE bool
	connectionSide(int aIndex) const
	{
		//if (aIndex < 0) printf("connectionSide()\n");
		return CONNECTION_VERTEX & connectionIndices[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow
	sourceTLinkCapacity(int aIndex) const
	{
		//if (aIndex < 0) printf("sourceTLinkCapacity()\n");
		return mSourceTLinks[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow
	sinkTLinkCapacity(int aIndex) const
	{
		//if (aIndex < 0) printf("sinkTLinkCapacity()\n");
		return mSinkTLinks[aIndex];
	}

	CUGIP_DECL_DEVICE EdgeResidualsRecord<TFlow> &
	residuals(int aIndex)
	{
		//if (aIndex < 0 || aIndex >= mEdgeCount) {printf("residuals() %d %d\n", aIndex, mEdgeCount); return mResiduals[0]; }
		//assert(aIndex >= 0);
		//assert(aIndex < mEdgeCount);
		return mResiduals[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow &
	sinkFlow(int aVertexId)
	{
		return mSinkFlow[aVertexId];
	}


	int mVertexCount;
	int mEdgeCount;

	TFlow *vertexExcess; // n
	int *labels; // n
	int *neighbors; // n

	TFlow *mSourceTLinks; // n
	TFlow *mSinkTLinks; // n

	int *secondVertices; // 2 * m
	int *connectionIndices; // 2 * m
	EdgeResidualsRecord<TFlow> *mResiduals; // m

	TFlow *mSinkFlow; // n

};

template<typename TFlow, typename TFunctor>
CUGIP_GLOBAL void
forEachVertexKernel(GraphCutData<TFlow> aGraph, TFunctor aFunctor)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int index = blockId * blockDim.x + threadIdx.x;

	while (index < aGraph.vertexCount()) {
		aFunctor(index, aGraph);
		index += blockDim.x * gridDim.x;
	}
}

template<typename TFlow, typename TFunctor>
void
for_each_vertex(GraphCutData<TFlow> &aGraph, TFunctor aFunctor)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D((aGraph.vertexCount() + blockSize1D.x - 1) / (blockSize1D.x) , 1);

	forEachVertexKernel<TFlow, TFunctor><<<gridSize1D, blockSize1D>>>(aGraph, aFunctor);

	CUGIP_CHECK_ERROR_STATE("After forEachVertexKernel");
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
}


} //namespace cugip
