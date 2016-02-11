#pragma once

#include <cugip/advanced_operations/detail/edge_record.hpp>

namespace cugip {

enum class RelabelImplementation {
	Default,
	Naive,
	OptimizedNaive
};

enum class PreflowInitialization {
	Default,
	Push
};

enum class TLinkType: int {
	Source = 0,
	Sink = 1
};

template<
	int tThreadCount = 512,
	int tGranularity = 64,
	typename TEdgeCheck = EdgeReverseTraversable,
	TLinkType tStartTLinkType = TLinkType::Sink>
struct RelabelPolicy {
	//static constexpr RelabelImplementation cRelabelImplementation = RelabelImplementation::Naive;
	//static constexpr RelabelImplementation cRelabelImplementation = RelabelImplementation::OptimizedNaive;
	static constexpr RelabelImplementation cRelabelImplementation = RelabelImplementation::Default;
	static constexpr TLinkType cStartTLinkType = tStartTLinkType;
	enum {
		INVALID_LABEL = 1 << 31,
		THREADS = tThreadCount,
		SCRATCH_ELEMENTS = THREADS,
		TILE_SIZE = THREADS,
		SCHEDULE_GRANULARITY = tGranularity,
		MULTI_LEVEL_LIMIT = 1024,
		MULTI_LEVEL_COUNT_LIMIT = 1000,
	};
	struct SharedMemoryData {
		//cub::BlockScan<int, BLOCK_SIZE> temp_storage;
		int offsetScratch[SCRATCH_ELEMENTS];
		//int incomming[SCRATCH_ELEMENTS];
	};

	RelabelPolicy(int aMaxLevels = (1 << 30))
		: maxAssignedLevels(aMaxLevels)
	{}

	CUGIP_DECL_HYBRID int
	maxLevels() const
	{
		return maxAssignedLevels; //TODO
	}
	TEdgeCheck edgeTraversalCheck;

	int maxAssignedLevels;
};

struct PushPolicy {
	enum {
		PUSH_ITERATION_ALGORITHM = 0 //TODO - define constants for each variant
	};
};

struct GraphCutPolicy
{
	static constexpr PreflowInitialization cPreflowInitialization = PreflowInitialization::Default;
	//static constexpr PreflowInitialization cPreflowInitialization = PreflowInitialization::Push;

	typedef RelabelPolicy<512, 64, EdgeReverseTraversable> RelabelPolicy;
	typedef PushPolicy PushPolicy;
};

} // namespace cugip
