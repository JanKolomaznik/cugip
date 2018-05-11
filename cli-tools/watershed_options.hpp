#pragma once

enum class WatershedVariant {
	DistanceBased,
	DistanceBasedAsync,
	DistanceBasedAsyncLimited,
	SteepestDescentSimple,
	SteepestDescentSimpleAsync,
	SteepestDescentGlobalState,
	SteepestDescentPointer,
	SteepestDescentPointerTwoPhase,
};

struct WatershedOptions
{
	WatershedVariant wshedVariant = WatershedVariant::SteepestDescentSimple;
	bool useUnifiedMemory = true;
};
