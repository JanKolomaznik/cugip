#pragma once

enum class WatershedVariant {
	DistanceBased,
	SteepestDescentSimple,
	SteepestDescentGlobalState
};

struct WatershedOptions
{
	WatershedVariant wshedVariant = WatershedVariant::SteepestDescentSimple;
};
