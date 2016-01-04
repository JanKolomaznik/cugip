#pragma once

#include <cugip/detail/defines.hpp>
#include <cugip/math.hpp>

namespace cugip {

typedef unsigned NodeId;
typedef unsigned long CombinedNodeId;
struct EdgeRecord
{
	CUGIP_DECL_HYBRID
	EdgeRecord( NodeId aFirst, NodeId aSecond )
	{
		first = min( aFirst, aSecond );
		second = max( aFirst, aSecond );
	}
	CUGIP_DECL_HYBRID
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

	CUGIP_DECL_HYBRID const float &
	getResidual( bool aFirst ) const
	{
		return aFirst ? residuals[0] : residuals[1];
	}
	TFlow residuals[2];
};


struct EdgeReverseTraversable
{
	template<typename TResiduals>
	CUGIP_DECL_DEVICE bool
	invoke(bool aConnectionSide, const TResiduals &aResiduals) const
	{
		auto residual = aResiduals.getResidual(!aConnectionSide);
		return residual > 0.0f;
	}
};

struct EdgeForwardTraversable
{
	template<typename TResiduals>
	CUGIP_DECL_DEVICE bool
	invoke(bool aConnectionSide, const TResiduals &aResiduals) const
	{
		auto residual = aResiduals.getResidual(aConnectionSide);
		return residual > 0.1f;
	}
};

}  // namespace cugip
