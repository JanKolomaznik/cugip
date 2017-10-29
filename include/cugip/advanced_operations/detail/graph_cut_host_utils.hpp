#pragma once

#include <vector>
#include <cugip/advanced_operations/detail/edge_record.hpp>

namespace cugip {

template<typename TWeight>
struct GraphData
{
	void
	setVertexCount(int aVertexCount)
	{
		tlinksSource.resize(aVertexCount);
		tlinksSink.resize(aVertexCount);
	}

	void
	setTWeights(int aVertex, TWeight aSourceWeight, TWeight aSinkWeight)
	{
		tlinksSource[aVertex] = aSourceWeight;
		tlinksSink[aVertex] = aSinkWeight;
	}

	void
	addEdge(int aVertex1, int aVertex2, TWeight aForwardWeight, TWeight aBackwardWeight)
	{
		edges.emplace_back(aVertex1, aVertex2);
		weights.push_back(aForwardWeight);
		weightsBackward.push_back(aBackwardWeight);
	}

	void
	reserve(int aVertexCount, int aEdgeCount)
	{
		tlinksSource.reserve(aVertexCount);
		tlinksSink.reserve(aVertexCount);

		edges.reserve(aEdgeCount);
		weights.reserve(aEdgeCount);
		weightsBackward.reserve(aEdgeCount);
	}
	std::vector<TWeight> tlinksSource;
	std::vector<TWeight> tlinksSink;
	std::vector<EdgeRecord> edges;
	std::vector<TWeight> weights;
	std::vector<TWeight> weightsBackward;
};

}  // namespace cugip
