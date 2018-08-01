
#include <cugip/host_image_view.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/region.hpp>
#include <cugip/for_each.hpp>

#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>

#include <unordered_map>

struct PairHash {
public:
	template <typename T, typename U>
	std::size_t operator()(const std::pair<T, U> &x) const
	{
		return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
	}
};

struct NodeStatistics
{
	NodeStatistics()
		: mCount(0)
		, mSum(0.0f)
		, mSquareSum(0.0f)
	{}

	void
	addValue(float aValue)
	{
		++mCount;
		mSum += aValue;
		mSquareSum += aValue * aValue;
	}

	void
	consolidate()
	{

	}

	int mCount;
	float mSum;
	float mSquareSum;
};

struct EdgeStatistics
{
	EdgeStatistics()
		: mCount(0)
	{}

	void
	add()
	{
		++mCount;
	}

	int mCount;
};

struct GraphDescription
{
	void
	addNodeValue(int aNodeId, float aValue)
	{
		mNodes[aNodeId].addValue(aValue);
	}

	void
	addNodeConnection(int aNodeIdFirst, int aNodeIdSecond)
	{
		auto edge = std::make_pair(std::min(aNodeIdFirst, aNodeIdSecond), std::max(aNodeIdFirst, aNodeIdSecond));
		mEdges[edge].add();
	}

	void
	consolidate()
	{
		for (auto &node : mNodes) {
			node.second.consolidate();
		}
	}

	std::unordered_map<int, NodeStatistics> mNodes;
	std::unordered_map<std::pair<int, int>, EdgeStatistics, PairHash> mEdges;
};

void
regionsNeighborhood(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const int, 3> aRegions)
{
	using namespace cugip;
	VonNeumannNeighborhood<3> neighborhood;

	GraphDescription graph;

	auto size = aData.dimensions();
	region<3> imageRegion{ Int3(), size };
	for_each(
		imageRegion,
		[&](const Int3 &coordinate) {
			auto value = aData[coordinate];
			auto id = aRegions[coordinate];
			graph.addNodeValue(id, value);
			for (int n = 1; n < (neighborhood.size() + 1) / 2; ++n) {
				Int3 neighbor = coordinate + neighborhood.offset(n);
				if (isInsideRegion(aData.dimensions(), neighbor)) {
					auto id2 = aRegions[neighbor];
					if (id2 != id) {
						graph.addNodeConnection(id, id2);
					}
				}
			}

		});

}
