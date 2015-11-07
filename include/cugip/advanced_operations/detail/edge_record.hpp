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


}  // namespace cugip
