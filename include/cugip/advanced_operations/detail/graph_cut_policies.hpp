#pragma once

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

struct GraphCutPolicy
{
	static constexpr PreflowInitialization cPreflowInitialization = PreflowInitialization::Default;
	template<int tThreadCount = 512, int tGranularity = 64>
	struct RelabelPolicy {
		//static constexpr RelabelImplementation cRelabelImplementation = RelabelImplementation::Naive;
		static constexpr RelabelImplementation cRelabelImplementation = RelabelImplementation::OptimizedNaive;
		//static constexpr RelabelImplementation cRelabelImplementation = RelabelImplementation::Default;
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
	};
	struct PushPolicy {
		enum {
			PUSH_ITERATION_ALGORITHM = 0 //TODO - define constants for each variant
		};
	};
};

} // namespace cugip
