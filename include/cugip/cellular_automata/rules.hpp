#pragma once

#include <type_traits>

#include <cugip/cuda_utils.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/cellular_automata/global_state.hpp>
#include <cugip/equivalence_management.hpp>


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
	preprocess(TView aView) {}

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

template<typename TId=int>
struct LocalMinimaEquivalenceGlobalState
{
	struct Relabel
	{
		template<typename TValue>
		CUGIP_DECL_DEVICE
		TValue operator()(TValue aLabel) const
		{
			/*if (aLabel >= manager.mSize) {
				printf("AAAAAAAAAAAAAAAA\n");
				return 0;
			}*/
			get<1>(aLabel) =  manager.get(get<1>(aLabel));
			return aLabel;
		}

		EquivalenceManager<TId> manager;
	};

	void
	initialize(){
		manager.initialize();
		mDeviceFlag.reset_host();
	}

	template<typename TView>
	void
	preprocess(TView aView)
	{
		mDeviceFlag.reset_host();
	}

	template<typename TView>
	void
	postprocess(TView aView)
	{
		manager.compaction();
		for_each(aView, Relabel{ manager });
	}

	CUGIP_DECL_DEVICE
	void
	signal()
	{
		mDeviceFlag.set_device();
	}

	bool is_finished()
	{
		return !mDeviceFlag.check_host();
	}

	EquivalenceManager<TId> manager;
	device_flag_view mDeviceFlag;
};

struct LocalMinimaConnectedComponentRule
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	template<typename TNeighborhood, typename TId>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, LocalMinimaEquivalenceGlobalState<TId> aEquivalence) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		auto value = aNeighborhood[0];
		auto minLabel = get<1>(value);
		if (minLabel) {
			for (int i = 1; i < aNeighborhood.size(); ++i) {
				//printf("Value %d - %d: neighbor %d: %d = %d, [%d, %d]\n", get<0>(value), get<1>(value), i, get<0>(aNeighborhood[i]), get<1>(aNeighborhood[i]), threadIdx.x, threadIdx.y);
				//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
				if (get<1>(aNeighborhood[i]) > 0 && get<1>(aNeighborhood[i]) < minLabel) {
					minLabel = get<1>(aNeighborhood[i]);
				}
				if (get<0>(aNeighborhood[i]) < get<0>(value)) {
					minLabel = 0;
				}
				if (get<1>(aNeighborhood[i]) == 0 && get<0>(value) == get<0>(aNeighborhood[i])) {
					//printf("Value %d - %d: neighbor %d: %d = %d\n", get<0>(value), get<1>(value), i, get<0>(aNeighborhood[i]), get<1>(aNeighborhood[i]));
					minLabel = 0;
				}
			}
			if (minLabel < get<1>(value)) {
				aEquivalence.manager.merge(minLabel, get<1>(value));
				//printf("%d %d\n", minValue, get<1>(value));
				get<1>(value) = minLabel;
				aEquivalence.signal();
			}
		}
		return value;
	}
};

/*struct WatershedConvergenceGlobalState
{
	void
	initialize(){
		mDeviceFlag.reset_host();
	}

	template<typename TView>
	void
	postprocess(TView aView)
	{
	}

	CUGIP_DECL_DEVICE
	void
	signal()
	{
		mDeviceFlag.set_device();
	}
	device_flag_view mDeviceFlag;
};*/

struct WatershedRule
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;
	//TODO - global state by reference
	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, ConvergenceFlag aConvergenceState) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label, distance
		auto value = aNeighborhood[0];
		int index = -1;
		auto minValue = get<2>(value);// - get<0>(value);
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			auto distance = get<2>(aNeighborhood[i]);
			if (get<1>(aNeighborhood[i]) != 0 && distance < minValue) {
				index = i;
				minValue = distance;
			}
			//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
		}
		auto newValue = minValue + get<0>(value);
		if (index != -1 && newValue < get<2>(value)) {
			//printf("aaa %d - %d %f\n", get<1>(value), get<1>(aNeighborhood[index]), newValue);
			get<1>(value) = get<1>(aNeighborhood[index]);
			get<2>(value) = newValue;//get<2>(aNeighborhood[index]) + get<0>(value);
			aConvergenceState.signal();
		}
		return value;
	}
};

typedef LocalMinimaEquivalenceGlobalState<int> Watershed2EquivalenceGlobalState;


/*struct Watershed2EquivalenceGlobalState
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
};*/

struct Watershed2RuleBase
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	int findNeighborForMerge(TNeighborhood aNeighborhood)
	{
		//input, label, has smaller neighbor
		auto value = aNeighborhood[0];
		int index = -1;
		auto minValue = get<0>(value);
		int plateauIndex = -1;
		int plateauLabel = get<1>(value);

		//printf("%d - %d; ", get<0>(value), get<1>(value));
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			auto current = get<0>(aNeighborhood[i]);
			//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, get<1>(aNeighborhood[0]), get<1>(aNeighborhood[i]));
			if (current < minValue) {
				index = i;
				minValue = current;
			} else {
				if (
					current == get<0>(value) &&
					get<2>(aNeighborhood[i]) == 0 &&
					get<1>(aNeighborhood[i]) < plateauLabel
				) {
					plateauIndex = i;
					plateauLabel = get<1>(aNeighborhood[i]);
				}
			}
		}
		return index != -1 ? index : plateauIndex;
	}
};

struct Watershed2Rule : Watershed2RuleBase
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	//TODO - global state by reference
	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label, has smaller neighbor
		auto value = aNeighborhood[0];
		int index = findNeighborForMerge(aNeighborhood);
		if (index != -1 && get<1>(value) != get<1>(aNeighborhood[index])) {
			get<1>(value) = get<1>(aNeighborhood[index]);
		}
		return value;
	}
};

struct Watershed2GlobalStateRule : Watershed2RuleBase
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	//TODO - global state by reference
	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, Watershed2EquivalenceGlobalState aEquivalence) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label, has smaller neighbor
		auto value = aNeighborhood[0];
		int index = findNeighborForMerge(aNeighborhood);
		if (index != -1 && get<1>(value) != get<1>(aNeighborhood[index])) {
				aEquivalence.manager.merge(get<1>(value), get<1>(aNeighborhood[index]));
				aEquivalence.signal();
				if (get<1>(value) > get<1>(aNeighborhood[index])) {
					get<1>(value) = get<1>(aNeighborhood[index]);
				}
		}
		return value;
	}
};

struct Watershed3Rule
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	//TODO - global state by reference
	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, Watershed2EquivalenceGlobalState aEquivalence) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label
		auto value = aNeighborhood[0];
		int index = -1;
		auto minValue = get<0>(value);
		auto minLabel = get<1>(value);
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			auto current = get<0>(aNeighborhood[i]);
			auto currentLabel = get<1>(aNeighborhood[i]);
			//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
			if (current <= minValue && currentLabel < minLabel) {
				index = i;
				minValue = current;
			}
		}
		if (index != -1) {
			aEquivalence.manager.merge(get<1>(value), get<1>(aNeighborhood[index]));
			aEquivalence.signal();
		}
		return value;
	}
};

struct ReactionDiffusionRule
{
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	template<typename TValue>
	CUGIP_DECL_DEVICE
	float f(const TValue &aValue) const {
		//return - aValue[0]*sqr(aValue[1]);
		return alpha * aValue[0] - aValue[0]*sqr(aValue[1]);
	}

	template<typename TValue>
	CUGIP_DECL_DEVICE
	float g(const TValue &aValue) const {
		return aValue[0]*sqr(aValue[1]) - beta * aValue[1];
		//return aValue[0]*sqr(aValue[1]) - ((alpha + beta) * aValue[0]) / (1.0f + aValue[0] + aValue[1]);
	}


	//TODO - global state by reference
	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		auto value = aNeighborhood[0];
		auto laplace = -1.0f * (aNeighborhood.size() - 1) * value;
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			laplace += aNeighborhood[i];
		}
		/*value += product(k, sum);
		//return value;
		value[0] += f(aNeighborhood[0]);
		value[1] += g(aNeighborhood[0]);*/
		auto delta = product(k, laplace);
		delta[0] += f(value);
		delta[1] += g(value);
		return value + timestep * delta;
	}
	float timestep = 0.2;
	float alpha = 0.1f;
	float beta = 0.1f;
	vect2f_t k = vect2f_t(0.64f, 0.32f);
};



} // namespace cugip
