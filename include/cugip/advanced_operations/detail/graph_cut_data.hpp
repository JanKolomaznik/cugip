#pragma once


namespace cugip {

enum {
	CONNECTION_VERTEX = 1 << 31,
	CONNECTION_INDEX_MASK = ~CONNECTION_VERTEX
};

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

template<typename TFlow>
struct GraphCutData
{
	CUGIP_DECL_DEVICE int
	neighborCount(int aVertexId)
	{
		if (aVertexId < 0) printf("neighborCount()\n");
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
		if (aVertexId < 0) printf("label()\n");
		return labels[aVertexId];
	}

	CUGIP_DECL_HYBRID int
	vertexCount()
	{
		return mVertexCount;
	}

	CUGIP_DECL_DEVICE int
	firstNeighborIndex(int aVertexId)
	{
		if (aVertexId < 0) printf("firstNeighborIndex()\n");
		return neighbors[aVertexId];
	}

	CUGIP_DECL_DEVICE int
	secondVertex(int aIndex)
	{
		if (aIndex < 0) printf("secondVertex()\n");
		return secondVertices[aIndex];
	}

	CUGIP_DECL_DEVICE int
	connectionIndex(int aIndex)
	{
		if (aIndex < 0) printf("connectionIndex()\n");
		return CONNECTION_INDEX_MASK & connectionIndices[aIndex];
	}

	CUGIP_DECL_DEVICE bool
	connectionSide(int aIndex)
	{
		if (aIndex < 0) printf("connectionSide()\n");
		return CONNECTION_VERTEX & connectionIndices[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow
	sourceTLinkCapacity(int aIndex)
	{
		if (aIndex < 0) printf("sourceTLinkCapacity()\n");
		return mSourceTLinks[aIndex];
	}

	CUGIP_DECL_DEVICE TFlow
	sinkTLinkCapacity(int aIndex)
	{
		if (aIndex < 0) printf("sinkTLinkCapacity()\n");
		return mSinkTLinks[aIndex];
	}

	CUGIP_DECL_DEVICE EdgeResidualsRecord<TFlow> &
	residuals(int aIndex)
	{
		if (aIndex < 0) printf("residuals()\n");
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

	TFlow *mSinkFlow;

};

} //namespace cugip
