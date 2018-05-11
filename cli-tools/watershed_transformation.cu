#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

#include <chrono>
#include <thread>

#include <cugip/image.hpp>
//#include <cugip/memory_view.hpp>
//#include <cugip/memory.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/unified_image_view.hpp>
#include <cugip/cellular_automata/cellular_automata.hpp>
#include <cugip/cellular_automata/async_cellular_automata.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/view_arithmetics.hpp>
#include <cugip/image_dumping.hpp>

#include <thrust/device_vector.h>

#include "watershed_options.hpp"

#include <cugip/timers.hpp>

using namespace cugip;

template<bool tUseUnifiedMemory = false>
struct MemoryTraits
{
	template<typename TElement, int tDimension>
	using image = device_image<TElement, tDimension>;
};

template<>
struct MemoryTraits<true>
{
	template<typename TElement, int tDimension>
	using image = unified_image<TElement, tDimension>;
};

template<typename TElement, int tDimension, bool tUseUnifiedMemory>
using image = typename MemoryTraits<tUseUnifiedMemory>::image<TElement, tDimension>;


struct InitWatershed
{
	template<typename TLabel>
	CUGIP_DECL_HYBRID
	Tuple<float, TLabel, float>
	operator()(float aGradient, TLabel aLocalMinimum) const
	{
		return Tuple<float, TLabel, float>(aGradient, aLocalMinimum, aLocalMinimum > 0 ? 0 : 1.0e15);
	}
};

struct ZipGradientAndLabel
{
	template<typename TLabel>
	CUGIP_DECL_HYBRID
	Tuple<float, TLabel>
	operator()(float aGradient, TLabel aLocalMinimum) const
	{
		return Tuple<float, TLabel>(aGradient, aLocalMinimum);
	}
};

struct BlockConvergenceFlagView
{
	template<typename TGlobalState>
	CUGIP_DECL_DEVICE
	void update_global(TGlobalState &aGlobalState)
	{
		//TODO
		if (*is_signaled_ptr || *iteration_ptr > 1) {
			aGlobalState.signal();
		}
	}

	template<typename TGlobalState>
	CUGIP_DECL_DEVICE
	void update_global2(TGlobalState &aGlobalState)
	{
		update_global(aGlobalState);
	}

	CUGIP_DECL_DEVICE
	void signal()
	{
		*is_signaled_ptr = true;
	}
	bool *is_signaled_ptr;
	int *iteration_ptr;
};

template<int tIterationLimit = 10000>
struct BlockConvergenceFlag: public BlockConvergenceFlagView
{
	CUGIP_DECL_DEVICE
	void initialize()
	{
		this->is_signaled_ptr = &is_signaled;
		this->iteration_ptr = &iteration;
		iteration = 0;
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
		return !is_signaled || iteration > tIterationLimit;
	}

	CUGIP_DECL_DEVICE
	BlockConvergenceFlagView view()
	{
		return static_cast<BlockConvergenceFlagView>(*this);
	}
	bool is_signaled;
	int iteration;
};


struct WatershedByPointer : WatershedSteepestDescentRuleBase
{
	enum Indices {
		cValue = 0,
		cLabel,
	};
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	struct HelperState {
		CUGIP_DECL_DEVICE
		void signal(){}
	};

	CUGIP_DECL_HYBRID
	static bool IsHelperState(const HelperState &) {
		return true;
	}

	template<typename TConvergenceFlag>
	CUGIP_DECL_HYBRID
	static bool IsHelperState(const TConvergenceFlag &) {
		return false;
	}

	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		return (*this)(aIteration, aNeighborhood, HelperState{});
	}
	//TODO - global state by reference
	template<typename TNeighborhood, typename TConvergenceFlag>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, TConvergenceFlag aConvergenceState) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label
		auto gridView = aNeighborhood.locator().view();
		auto value = aNeighborhood[0];
		if (get<cLabel>(value) < 0) {
			auto position = index_from_linear_access_index(gridView, (-1 * get<cLabel>(value)) - 1);
			auto newValue = gridView[position];
			/*int currentLabel = -1 * (1 + get_linear_access_index(gridView.dimensions(), aNeighborhood.locator().coords()));
			if (aIteration > 30) {
				printf("AAAAAAAAAAA %d %d, %d\n", get<1>(value), get<1>(newValue), currentLabel);
			}*/
			if (get<cLabel>(value) != get<cLabel>(newValue)) {
				get<cLabel>(value) = get<cLabel>(newValue);
				aConvergenceState.signal();
			}
		} else {
			if (get<cLabel>(value) == 0) {
				int index = -1;
				auto minValue = get<cValue>(value);
				for (int i = 1; i < aNeighborhood.size(); ++i) {
					auto current = get<cValue>(aNeighborhood[i]);
					//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, get<1>(aNeighborhood[0]), get<1>(aNeighborhood[i]));
					if (current <= minValue && aNeighborhood.is_inside_valid_region(i)) {
						index = i;
						minValue = current;
					}
				}
				if (index != -1) {
					//auto tmpIndex = aNeighborhood.view_index(index);
					Int3 tmpIndex = currentThreadIndex();//aNeighborhood.mLocator.coords();
					Int3 tmpIndex2 = currentBlockIndex();//aNeighborhood.offset(index);
					int neighborIndex = get_linear_access_index(gridView.dimensions(), aNeighborhood.view_index(index));
					get<cLabel>(value) = -1 * (1 + neighborIndex);
					//if (!IsHelperState(aConvergenceState))
					/*if (tmpIndex2 == Int3(0, 1, 0) && tmpIndex == Int3(0, 1, 2))
						//printf("%d %d - %d val = %d -> {%d, %d}; %d - [%d, %d, %d], [%d, %d, %d]\n",
						printf("%d ; %d - [%d, %d, %d], [%d, %d, %d] %f, %f\n",
							index,
							neighborIndex,
							tmpIndex[0], tmpIndex[1], tmpIndex[2],
							tmpIndex2[0], tmpIndex2[1], tmpIndex2[2],
							get<cValue>(value),
							get<cValue>(aNeighborhood[index])
						);*/
					aConvergenceState.signal();
				}
			}
		}
		return value;
	}
};



template<int tDimension, typename TGradientView, typename TLabelView>
void initLocalMinimaLabels(
	TGradientView deviceGradient,
	TLabelView labels,
	AggregatingTimerSet<2, int> &timer,
	device_flag &convergenceFlag
)
{
	typedef typename TLabelView::value_type Label;
	typedef Tuple<float, Label> Value2;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value2, tDimension>,
			VonNeumannNeighborhood<tDimension>,
			LocalMinimaConnectedComponentRule,
			LocalMinimaEquivalenceGlobalState<Label>> LocalMinimaAutomaton;
	auto localMinima = unaryOperatorOnLocator(deviceGradient, LocalMinimumLabel<Label>());

	LocalMinimaEquivalenceGlobalState<Label> globalState;

	int64_t count = elementCount(deviceGradient) + 1;
	/*thrust::device_vector<Label> buffer;
	buffer.resize(count);*/

	using deleted_unique_ptr = std::unique_ptr<Label, void(*)(Label*)>;
	Label *ptr = nullptr;
	CUGIP_CHECK_RESULT(cudaMallocManaged(&ptr, sizeof(Label) * count));
	auto pointer = deleted_unique_ptr(ptr, [](Label *buffer) { cudaFree(buffer); });

	globalState.manager = EquivalenceManager<Label>(pointer.get(), count);
	//globalState.manager = EquivalenceManager<Label>(thrust::raw_pointer_cast(&buffer[0]), count);
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	LocalMinimaAutomaton localMinimumAutomaton;
	localMinimumAutomaton.initialize(
		nAryOperator(ZipGradientAndLabel(), deviceGradient, localMinima),
		globalState);

	int iteration = 0;
	do {
		auto interval = timer.start(0, iteration++);
		localMinimumAutomaton.iterate(1);
	} while (!globalState.is_finished());
	//localMinimumAutomaton.iterate(100);
	copy(getDimension(localMinimumAutomaton.getCurrentState(), IntValue<1>()), labels);
}

template<int tDimension, typename TLabel, bool tUseUnifiedMemory>
void distanceBasedWShed(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, TLabel, float> Value;
	image<float, tDimension, tUseUnifiedMemory> deviceGradient(aInput.dimensions());
	image<TLabel, tDimension, tUseUnifiedMemory> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));
	device_flag convergenceFlag;
	AggregatingTimerSet<2, int> timer;
	initLocalMinimaLabels<tDimension>(const_view(deviceGradient), view(labels), timer, convergenceFlag);

	auto wshed = nAryOperator(InitWatershed(), const_view(deviceGradient), const_view(labels));

	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedRule,
			ConvergenceFlag> WatershedAutomaton;
	ConvergenceFlag convergenceGlobalState;
	convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

	WatershedAutomaton automaton;
	automaton.initialize(wshed, convergenceGlobalState);

	int iteration = 0;
	do {
		auto interval = timer.start(1, iteration++);
		automaton.iterate(1);
	} while (!convergenceGlobalState.is_finished());

	std::cout << timer.createReport({"Local minima search", "Wshed iterations"});
	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labels));
	copy(const_view(labels), aOutput);
}

template<int tDimension, int tIterationLimit, typename TLabel, bool tUseUnifiedMemory>
void distanceBasedWShedAsync(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, TLabel, float> Value;
	image<float, tDimension, tUseUnifiedMemory> deviceGradient(aInput.dimensions());
	image<TLabel, tDimension, tUseUnifiedMemory> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));
	device_flag convergenceFlag;
	AggregatingTimerSet<2, int> timer;
	initLocalMinimaLabels<tDimension>(const_view(deviceGradient), view(labels), timer, convergenceFlag);

	auto wshed = nAryOperator(InitWatershed(), const_view(deviceGradient), const_view(labels));

	typedef AsyncCellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedRule,
			ConvergenceFlag,
			BlockConvergenceFlag<tIterationLimit>> WatershedAutomaton;
	ConvergenceFlag convergenceGlobalState;
	convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

	WatershedAutomaton automaton;
	automaton.initialize(wshed, convergenceGlobalState);

	int iteration = 0;
	do {
		auto interval = timer.start(1, iteration++);
		automaton.iterate(1);
	} while (!convergenceGlobalState.is_finished());

	std::cout << timer.createReport({"Local minima search", "Wshed iterations"});
	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labels));
	copy(const_view(labels), aOutput);
}


template<int tDimension, bool tUseUnifiedMemory, typename TView>
void lowerCompletion(TView aInput)
{
	//lower completion
	image<float, tDimension, tUseUnifiedMemory> tmp(aInput.dimensions());
	transform_locator(aInput, view(tmp), HandlePlateauBorder());
	copy(const_view(tmp), aInput);
}

template<int tDimension, typename TLabel, bool tUseUnifiedMemory>
void steepestDescentWShedSimple(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, TLabel> Value;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedSteepestDescentRule,
			ConvergenceFlag> WatershedAutomaton;
	image<float, tDimension, tUseUnifiedMemory> deviceGradient(aInput.dimensions());
	image<TLabel, tDimension, tUseUnifiedMemory> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));

	lowerCompletion<tDimension, tUseUnifiedMemory>(view(deviceGradient));
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, LinearAccessIndex<TLabel>>(aInput.dimensions()));

	device_flag convergenceFlag;
	ConvergenceFlag convergenceGlobalState;
	convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

	WatershedAutomaton automaton;
	automaton.initialize(wshed, convergenceGlobalState);

	AggregatingTimerSet<1, int> timer;
	int iteration = automaton.run_until_convergence(
		[&]{
			return convergenceGlobalState.is_finished();
		});
	/*do {
		auto interval = timer.start(0, iteration++);
		automaton.iterate(1);
	} while (!convergenceGlobalState.is_finished());*/

	std::cout << timer.createReport({"Wshed iterations"});
	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labels));
	copy(const_view(labels), aOutput);
}

template<int tDimension, typename TLabel, bool tUseUnifiedMemory>
void steepestDescentWShedPointer(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, TLabel> Value;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedByPointer,
			ConvergenceFlag> WatershedAutomaton;
	unified_image<float, tDimension> deviceGradient(aInput.dimensions());
	unified_image<TLabel, tDimension> labels(aInput.dimensions());
	//device_image<float, tDimension> deviceGradient(aInput.dimensions());
	//device_image<TLabel, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));
	AggregatingTimerSet<2, int> timer;
	device_flag convergenceFlag;

	lowerCompletion<tDimension, tUseUnifiedMemory>(view(deviceGradient));
	initLocalMinimaLabels<tDimension>(const_view(deviceGradient), view(labels), timer, convergenceFlag);
	auto wshed = zipViews(const_view(deviceGradient), const_view(labels));

	ConvergenceFlag convergenceGlobalState;
	convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

	WatershedAutomaton automaton;
	automaton.initialize(wshed, convergenceGlobalState);

	int iteration = 0;
	do {
		auto interval = timer.start(1, iteration++);
		automaton.iterate(1);
	} while (!convergenceGlobalState.is_finished());

	std::cout << timer.createReport(std::array<std::string, 2>{"Local minima search", "Wshed iterations"} );
	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labels));
	copy(const_view(labels), aOutput);
}

template<int tDimension, typename TLabel, bool tUseUnifiedMemory>
void steepestDescentWShedPointerTwoPhase(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, TLabel> Value;
	unified_image<float, tDimension> deviceGradient(aInput.dimensions());
	unified_image<TLabel, tDimension> labels(aInput.dimensions());
	//device_image<float, tDimension> deviceGradient(aInput.dimensions());
	//device_image<TLabel, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));
	AggregatingTimerSet<2, int> timer;
	device_flag convergenceFlag;

	lowerCompletion<tDimension, tUseUnifiedMemory>(view(deviceGradient));
	initLocalMinimaLabels<tDimension>(const_view(deviceGradient), view(labels), timer, convergenceFlag);
	auto wshed = zipViews(const_view(deviceGradient), const_view(labels));

	unified_image<Value, tDimension> intermediateResult;
	{
		typedef CellularAutomaton<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedByPointer> WatershedAutomatonFirstPhase;

		WatershedAutomatonFirstPhase automaton;
		automaton.initialize(wshed);
		automaton.iterate(1);

		//dump_view(getDimension(automaton.getCurrentState(), IntValue<1>()), "./current_state1_");
		intermediateResult = automaton.moveCurrentState();
	}
	/*dump_view(getDimension(view(intermediateResult), IntValue<1>()), "./intermediate_result_");
	using namespace std::chrono_literals;
std::this_thread::sleep_for(2s);*/
	{
		typedef CellularAutomatonWithGlobalState<
				Grid<Value, tDimension>,
				NoNeighborhood<tDimension>,
				//MooreNeighborhood<tDimension>,
				WatershedByPointer,
				ConvergenceFlag> WatershedAutomatonSecondPhase;

		ConvergenceFlag convergenceGlobalState;
		convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

		WatershedAutomatonSecondPhase automaton;
		automaton.initialize(std::move(intermediateResult), convergenceGlobalState);

		//dump_view(getDimension(automaton.getCurrentState(), IntValue<1>()), "./current_state2_");
		int iteration = 0;
		do {
			auto interval = timer.start(1, iteration++);
			automaton.iterate(1);
		} while (!convergenceGlobalState.is_finished());
		std::cout << timer.createReport(std::array<std::string, 2>{"Local minima search", "Wshed iterations"} );
		auto state = automaton.getCurrentState();
		copy(getDimension(state, IntValue<1>()), view(labels));
		copy(const_view(labels), aOutput);
	}
}

template<int tDimension, typename TLabel, bool tUseUnifiedMemory>
void steepestDescentWShedSimpleAsync(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, TLabel> Value;
	typedef AsyncCellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedSteepestDescentRule,
			ConvergenceFlag,
			BlockConvergenceFlag<1000>> WatershedAutomaton;
	/*typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedSteepestDescentRule,
			ConvergenceFlag> WatershedAutomaton;*/
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<TLabel, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));

	lowerCompletion<tDimension, tUseUnifiedMemory>(view(deviceGradient));
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, LinearAccessIndex<TLabel>>(aInput.dimensions()));

	device_flag convergenceFlag;
	ConvergenceFlag convergenceGlobalState;
	convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

	WatershedAutomaton automaton;
	automaton.initialize(wshed, convergenceGlobalState);

	AggregatingTimerSet<1, int> timer;
	int iteration = 0;
	do {
		auto interval = timer.start(0, iteration++);
		automaton.iterate(1);
	} while (!convergenceGlobalState.is_finished());

	std::cout << timer.createReport({"Wshed iterations"});
	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labels));
	copy(const_view(labels), aOutput);
}

template<int tDimension, typename TLabel, bool tUseUnifiedMemory>
void steepestDescentWShedGlobalState(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, TLabel> Value;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedSteepestDescentGlobalStateRule,
			LocalMinimaEquivalenceGlobalState<TLabel>> WatershedAutomaton;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<TLabel, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));

	lowerCompletion<tDimension, tUseUnifiedMemory>(view(deviceGradient));
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, LinearAccessIndex<TLabel>>(aInput.dimensions()));

	device_flag convergenceFlag;
	LocalMinimaEquivalenceGlobalState<TLabel> globalState;

	thrust::device_vector<TLabel> buffer;
	buffer.resize(elementCount(aInput) + 1);
	globalState.manager = EquivalenceManager<TLabel>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	WatershedAutomaton automaton;
	automaton.initialize(wshed, globalState);

	AggregatingTimerSet<1, int> timer;
	int iteration = 0;
	do {
		auto interval = timer.start(0, iteration++);
		automaton.iterate(1);
	} while (!globalState.is_finished());

	std::cout << timer.createReport({"Wshed iterations"});
	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labels));
	copy(const_view(labels), aOutput);
}

template<int tDimension, typename TLabel>
void steepestDescentWShedAsyncGlobalState(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
}

template<int tDimension, typename TLabel>
void steepestDescentWShedPointers(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{

}

//******************************************************************************

template<typename TLabel, int tDimension, bool tUseUnifiedMemory>
void runWatershedTransformationDim(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<TLabel, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	switch (aOptions.wshedVariant) {
	case WatershedVariant::DistanceBased:
		distanceBasedWShed<tDimension, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::DistanceBasedAsync:
		distanceBasedWShedAsync<tDimension, 10000, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::DistanceBasedAsyncLimited:
		distanceBasedWShedAsync<tDimension, 2, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentSimple:
		steepestDescentWShedSimple<tDimension, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentSimpleAsync:
		steepestDescentWShedSimpleAsync<tDimension, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentGlobalState:
		steepestDescentWShedGlobalState<tDimension, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentPointer:
		steepestDescentWShedPointer<tDimension, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentPointerTwoPhase:
		steepestDescentWShedPointerTwoPhase<tDimension, TLabel, tUseUnifiedMemory>(aInput, aOutput, aOptions);
		break;
	default:
		std::cerr << "Unknown watershed variant\n";
	}
}

void runWatershedTransformation(
		const_host_image_view<const float, 2> aInput,
		host_image_view<int32_t, 2> aOutput,
		const WatershedOptions &aOptions)
{
	//runWatershedTransformationDim<2>(aInput, aOutput, aOptions);
}

void runWatershedTransformation(
		const_host_image_view<const float, 3> aInput,
		host_image_view<int32_t, 3> aOutput,
		const WatershedOptions &aOptions)
{
	if (aOptions.useUnifiedMemory) {
		runWatershedTransformationDim<int32_t, 3, true>(aInput, aOutput, aOptions);
	} else {
		runWatershedTransformationDim<int32_t, 3, false>(aInput, aOutput, aOptions);
	}
}

void runWatershedTransformation(
		const_host_image_view<const float, 2> aInput,
		host_image_view<int64_t, 2> aOutput,
		const WatershedOptions &aOptions)
{
	//runWatershedTransformationDim<2>(aInput, aOutput, aOptions);
}

void runWatershedTransformation(
		const_host_image_view<const float, 3> aInput,
		host_image_view<int64_t, 3> aOutput,
		const WatershedOptions &aOptions)
{
	if (aOptions.useUnifiedMemory) {
		runWatershedTransformationDim<int64_t, 3, true>(aInput, aOutput, aOptions);
	} else {
		runWatershedTransformationDim<int64_t, 3, false>(aInput, aOutput, aOptions);
	}
}
