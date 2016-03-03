#pragma once

#include <cugip/advanced_operations/detail/edge_record.hpp>
#include <cugip/advanced_operations/detail/graph_cut_policies.hpp>

namespace cugip {

__device__ __inline__ int32_t ld_gbl_cg(const int32_t *addr) {
  int return_value;
  asm("ld.global.cg.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

__device__ __inline__ double ld_gbl_cg(const double *addr) {
  double return_value;
  asm("ld.global.cg.f64 %0, [%1];" : "=d"(return_value) : "l"(addr));
  return return_value;
}

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
		//static_assert(sizeof(int) == 4, "Int is not 32bit");
		//if (aVertexId < 0 || mVertexCount < aVertexId) printf("firstNeighborIndex() %d, %d\n", aVertexId, mVertexCount);
		//return __ldg(neighbors + aVertexId);
		//return ld_gbl_cg(neighbors + aVertexId);
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
		//return mSourceTLinks[aIndex];
		return tLinkCapacity<TLinkType::Source>(aIndex);
	}

	CUGIP_DECL_DEVICE TFlow
	sinkTLinkCapacity(int aIndex) const
	{
		//if (aIndex < 0) printf("sinkTLinkCapacity()\n");
		//return mSinkTLinks[aIndex];
		return tLinkCapacity<TLinkType::Sink>(aIndex);
	}

	template<TLinkType tTLinkType>
	CUGIP_DECL_DEVICE TFlow
	tLinkCapacity(int aIndex) const
	{
		//if (aIndex < 0) printf("sourceTLinkCapacity()\n");
		return mTLinks[int(tTLinkType)][aIndex];
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

	//TFlow *mSourceTLinks; // n
	//TFlow *mSinkTLinks; // n

	TFlow *mTLinks[2]; //n

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
