#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

#include <cugip/image.hpp>
//#include <cugip/memory_view.hpp>
//#include <cugip/memory.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/cellular_automata/cellular_automata.hpp>
#include <cugip/cellular_automata/async_cellular_automata.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/view_arithmetics.hpp>

#include <thrust/device_vector.h>

#include "watershed_options.hpp"

#include <cugip/timers.hpp>

using namespace cugip;

struct BlockEquivalenceView
{
	CUGIP_DECL_DEVICE
	int updateMapping(int aVal, int aNewMap)
	{
		int idx = (aVal *11 + 17) % (32*6*6);
		while ((hash_map_ptr[idx][0] != 0 && hash_map_ptr[idx][0] != aVal)
			|| (hash_map_ptr[idx][0] == 0) && atomicCAS(&(hash_map_ptr[idx][0]), 0, aVal) != 0)
		{
			idx = (idx + 1) % (32*6*6);
		}
		/*if (hash_map_ptr[idx][1] == 0 && atomicCAS(&(hash_map_ptr[idx][1]), 0, aNewMap) == 0) {
			return aNewMap;
		}*/
		/*atomicMin(&(hash_map_ptr[idx][1]), aNewMap);*/
		hash_map_ptr[idx][1] = hash_map_ptr[idx][1] == 0 ? aNewMap : min(hash_map_ptr[idx][1], aNewMap);
		return hash_map_ptr[idx][1];
	}

	CUGIP_DECL_DEVICE
	Int2 &findMapping(int aVal)
	{
		int idx = (aVal *11 + 17) % (32*6*6);
		while (hash_map_ptr[idx][0] != 0 && hash_map_ptr[idx][0] != aVal) {
			idx = (idx + 1) % (32*6*6);
		}
		return hash_map_ptr[idx];
	}

	CUGIP_DECL_DEVICE
	int merge(int aFirst, int aSecond)
	{
		auto minVal = min(aFirst, aSecond);
		auto maxVal = max(aFirst, aSecond);

		minVal = updateMapping(maxVal, minVal);
		return minVal;
	}

	template<typename TGlobalState>
	CUGIP_DECL_DEVICE
	void update_global(TGlobalState &aGlobalState)
	{
		int index = threadOrderFromIndex();
		while (index < (32*6*6)) {
			if (hash_map_ptr[index][0] != 0) {
				aGlobalState.merge(hash_map_ptr[index][1], hash_map_ptr[index][0]);
			}
			index += currentBlockSize();
		}
	}

	CUGIP_DECL_DEVICE
	void signal()
	{
		*is_signaled_ptr = true;
	}
	bool *is_signaled_ptr;

	Int2 *hash_map_ptr;
};

struct BlockEquivalence: public BlockEquivalenceView
{
	CUGIP_DECL_DEVICE
	void initialize()
	{
		this->is_signaled_ptr = &is_signaled;
		this->hash_map_ptr = (Int2*)hash_map;
		iteration = 0;
		int index = threadOrderFromIndex();
		while (index < 2*(32*6*6)) {
			hash_map[index] = 0;
			index += currentBlockSize();
		}
	}

	CUGIP_DECL_DEVICE
	void preprocess()
	{
		if (is_in_thread(0,0,0)) {
			is_signaled = false;
			++iteration;
		}
	}

	CUGIP_DECL_DEVICE
	bool is_finished()
	{
		return !is_signaled/* || iteration > 300*/;
	}

	CUGIP_DECL_DEVICE
	BlockEquivalenceView view()
	{
		return static_cast<BlockEquivalenceView>(*this);
	}
	bool is_signaled;
	int iteration;

	//Int2 hash_map[32*6*6];
	int hash_map[2*32*6*6];
};


template<int tDimension>
void cclAsyncGlobalState(
		const_host_image_view<const int8_t, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput)

{
	//typedef Tuple<int8_t, int32_t> Value;
	typedef AsyncCellularAutomatonWithGlobalState<
			Grid<int32_t, tDimension>,
			MooreNeighborhood<tDimension>,
			ConnectedComponentLabelingRule2,
			LocalMinimaEquivalenceGlobalState<int32_t>,
			BlockEquivalence> CCLAutomaton;
	device_image<int8_t, tDimension> masks(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
	copy(aInput, view(masks));

	auto firstGeneration = maskView(UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()), const_view(masks), 0);

	device_flag convergenceFlag;
	LocalMinimaEquivalenceGlobalState<int32_t> globalState;

	thrust::device_vector<int32_t> buffer;
	buffer.resize(elementCount(aInput) + 1);
	globalState.manager = EquivalenceManager<int32_t>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	CCLAutomaton automaton;
	automaton.initialize(firstGeneration, globalState);

	AggregatingTimerSet<1, int> timer;
	int iteration = 0;
	do {
		auto interval = timer.start(0, iteration++);
		automaton.iterate(1);
	} while (!globalState.is_finished() && iteration < 1);

	std::cout << timer.createReport({"async CCL iterations"});
	auto state = automaton.getCurrentState();
	copy(state, view(labels));
	copy(const_view(labels), aOutput);
}

template<int tDimension>
void cclSyncGlobalState(
		const_host_image_view<const int8_t, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput)

{
	//typedef Tuple<int8_t, int32_t> Value;
	typedef CellularAutomatonWithGlobalState<
			Grid<int32_t, tDimension>,
			MooreNeighborhood<tDimension>,
			ConnectedComponentLabelingRule2,
			LocalMinimaEquivalenceGlobalState<int32_t>> CCLAutomaton;
	device_image<int8_t, tDimension> masks(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
	copy(aInput, view(masks));

	auto firstGeneration = maskView(UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()), const_view(masks), 0);

	device_flag convergenceFlag;
	LocalMinimaEquivalenceGlobalState<int32_t> globalState;

	thrust::device_vector<int32_t> buffer;
	buffer.resize(elementCount(aInput) + 1);
	globalState.manager = EquivalenceManager<int32_t>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	CCLAutomaton automaton;
	automaton.initialize(firstGeneration, globalState);

	AggregatingTimerSet<1, int> timer;
	int iteration = 0;
	do {
		auto interval = timer.start(0, iteration++);
		automaton.iterate(1);
	} while (!globalState.is_finished());

	std::cout << timer.createReport({"sync CCL iterations"});
	auto state = automaton.getCurrentState();
	copy(state, view(labels));
	copy(const_view(labels), aOutput);
}

//******************************************************************************

template<int tDimension>
void runConnectedComponentLabelingDim(
		const_host_image_view<const int8_t, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const std::string &aName)
{
	if (aName == "sync_gs") {
		cclSyncGlobalState<tDimension>(aInput, aOutput);
	} else if (aName == "async_gs") {
		cclAsyncGlobalState<tDimension>(aInput, aOutput);
	} else {
		std::cerr << "Unknown CCL variant\n";
	}
}

void runConnectedComponentLabeling(
		const_host_image_view<const int8_t, 2> aInput,
		host_image_view<int32_t, 2> aOutput,
		const std::string &aName)
{
	//runConnectedComponentLabelingDim<2>(aInput, aOutput, aName);
	throw "TODO";
}

void runConnectedComponentLabeling(
		const_host_image_view<const int8_t, 3> aInput,
		host_image_view<int32_t, 3> aOutput,
		const std::string &aName)
{
	runConnectedComponentLabelingDim<3>(aInput, aOutput, aName);
}
