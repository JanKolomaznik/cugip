#pragma once

#include <type_traits>

#include <cugip/cuda_utils.hpp>
#include <cugip/neighborhood.hpp>


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

struct LocalMinimaEquivalenceGlobalState
{
	struct Relabel
	{
		template<typename TValue>
		CUGIP_DECL_DEVICE
		TValue operator()(TValue aLabel) const
		{
			get<1>(aLabel) =  manager.get(get<1>(aLabel));
			return aLabel;
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

struct LocalMinimaConnectedComponentRule
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, LocalMinimaEquivalenceGlobalState aEquivalence) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		auto value = aNeighborhood[0];
		auto minValue = get<1>(value);
		if (get<1>(value)) {
			for (int i = 1; i < aNeighborhood.size(); ++i) {
				//printf("Value %d - %d: neighbor %d: %d = %d, [%d, %d]\n", get<0>(value), get<1>(value), i, get<0>(aNeighborhood[i]), get<1>(aNeighborhood[i]), threadIdx.x, threadIdx.y);
				//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
				if (get<1>(aNeighborhood[i]) > 0 && get<1>(aNeighborhood[i]) < minValue) {
					minValue = get<1>(aNeighborhood[i]);
				}
				if (get<0>(aNeighborhood[i]) < get<0>(value)) {
					minValue = 0;
				}
				if (get<1>(aNeighborhood[i]) == 0 && get<0>(value) == get<0>(aNeighborhood[i])) {
					//printf("Value %d - %d: neighbor %d: %d = %d\n", get<0>(value), get<1>(value), i, get<0>(aNeighborhood[i]), get<1>(aNeighborhood[i]));
					minValue = 0;
				}
			}
			if (minValue < get<1>(value)) {
				aEquivalence.manager.merge(minValue, get<1>(value));
				//printf("%d %d\n", minValue, get<1>(value));
				get<1>(value) = minValue;
			}
		}
		return value;
	}
};


struct WatershedRule
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label, distance
		auto value = aNeighborhood[0];
		int index = -1;
		auto minValue = get<2>(value) - get<0>(value);
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			auto distance = get<2>(aNeighborhood[i]);
			//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
			if (distance < minValue) {
				index = i;
				minValue = distance;
			}
		}
		if (index != -1) {
			get<1>(value) = get<1>(aNeighborhood[index]);
			get<2>(value) = get<2>(aNeighborhood[index]) + get<0>(value);
		}
		return value;
	}
};

struct Watershed2EquivalenceGlobalState
{
	struct Relabel
	{
		template<typename TValue>
		CUGIP_DECL_DEVICE
		TValue operator()(TValue aLabel) const
		{
			get<1>(aLabel) =  manager.get(get<1>(aLabel));
			return aLabel;
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

struct Watershed2Rule
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, Watershed2EquivalenceGlobalState aEquivalence) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label, distance
		auto value = aNeighborhood[0];
		int index = -1;
		auto minValue = get<0>(value);
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			auto current = get<0>(aNeighborhood[i]);
			//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
			if (current < minValue) {
				index = i;
				minValue = current;
			} else {
				if (current == minValue && current == get<0>(value) && get<2>(aNeighborhood[i]) == 0) {
					index = i;
					minValue = current;
				}
			}
		}
		if (index != -1) {
			if (minValue < get<0>(value)
				|| ((minValue == get<0>(value)) && (0 == get<2>(aNeighborhood[index]))))
			{
				if (get<1>(value) != get<1>(aNeighborhood[index])) {
					aEquivalence.manager.merge(get<1>(value), get<1>(aNeighborhood[index]));
				}
			}
		}
		return value;
	}
};



} // namespace cugip
