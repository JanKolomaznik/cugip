#pragma once

enum class WatershedVariant {
	DistanceBased,
	DistanceBasedAsync,
	DistanceBasedAsyncLimited,
	SteepestDescentSimple,
	SteepestDescentSimpleAsync,
	SteepestDescentGlobalState,
	SteepestDescentPointer,
};

struct WatershedOptions
{
	WatershedVariant wshedVariant = WatershedVariant::SteepestDescentSimple;
};
