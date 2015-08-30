#pragma once

#include <cugip/cuda_utils.hpp>
#include <cugip/cellular_automata/neighborhood.hpp>


namespace cugip {


struct ConwayRule
{
	template<typename TNeighborhood>
	CUGIP_DECL_HYBRID uint8_t
	operator()(int aIteration, TNeighborhood aNeighborhood) const
	{
		int sum = 0;
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			sum += aNeighborhood[i];
		}
		if (aNeighborhood[0]) {
			if (sum < 2 || sum > 3) {
				return 0;
			}
			return 1;
		} else {
			if (sum == 3) {
				return 1;
			}
			return 0;
		}
	}
};

struct ConnectedComponentLabelingRule
{
	template<typename TNeighborhood>
	CUGIP_DECL_HYBRID int
	operator()(int aIteration, TNeighborhood aNeighborhood) const
	{
		auto value = aNeighborhood[0];
		if (value) {
			for (int i = 1; i < aNeighborhood.size(); ++i) {
				//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
				if (aNeighborhood[i] > 0 && aNeighborhood[i] < value) {
					value = aNeighborhood[i];
				}
			}
		}
		return value;
	}
};


struct EquivalenceGlobalState
{
	struct Relabel
	{
		CUGIP_DECL_DEVICE
		int operator()(int aLabel) const
		{
			return manager.get(aLabel);
		}

		EquivalenceManager<int> manager;
	};

	void
	initialize(){
		manager.initialize();
	}

	template<typename TView>
	void
	postprocess(TView aView)
	{
		manager.compaction();
		for_each(aView, Relabel{ manager });
	}

	EquivalenceManager<int> manager;
};

struct ConnectedComponentLabelingRule2
{
	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE int
	operator()(int aIteration, TNeighborhood aNeighborhood, EquivalenceGlobalState aEquivalence) const
	{
		auto value = aNeighborhood[0];
		auto minValue = value;
		if (value) {
			for (int i = 1; i < aNeighborhood.size(); ++i) {
				//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
				if (aNeighborhood[i] > 0 && aNeighborhood[i] < minValue) {
					minValue = aNeighborhood[i];
				}
			}
			if (minValue < value) {
				aEquivalence.manager.merge(minValue, value);
			}
		}
		return value;
	}
};


} // namespace cugip
