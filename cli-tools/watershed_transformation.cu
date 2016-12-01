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


struct InitWatershed
{
	CUGIP_DECL_HYBRID
	Tuple<float, int32_t, float>
	operator()(float aGradient, int32_t aLocalMinimum) const
	{
		return Tuple<float, int32_t, float>(aGradient, aLocalMinimum, aLocalMinimum > 0 ? 0 : 1.0e15);
	}
};

struct ZipGradientAndLabel
{
	CUGIP_DECL_HYBRID
	Tuple<float, int32_t>
	operator()(float aGradient, int32_t aLocalMinimum) const
	{
		return Tuple<float, int32_t>(aGradient, aLocalMinimum);
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
	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	//TODO - global state by reference
	template<typename TNeighborhood, typename TConvergenceFlag>
	CUGIP_DECL_DEVICE
	auto operator()(int aIteration, TNeighborhood aNeighborhood, TConvergenceFlag aConvergenceState) -> remove_reference<decltype(aNeighborhood[0])> const
	{
		//input, label
		auto gridView = aNeighborhood.locator().view();
		auto value = aNeighborhood[0];
		if (get<1>(value) < 0) {
			auto position = index_from_linear_access_index(gridView, (-1 * get<1>(value)) - 1);
			auto newValue = gridView[position];
			/*int currentLabel = -1 * (1 + get_linear_access_index(gridView.dimensions(), aNeighborhood.locator().coords()));
			if (aIteration > 30) {
				printf("AAAAAAAAAAA %d %d, %d\n", get<1>(value), get<1>(newValue), currentLabel);
			}*/
			if (get<1>(value) != get<1>(newValue)) {
				get<1>(value) = get<1>(newValue);
				aConvergenceState.signal();
			}
		} else {
			if (get<1>(value) == 0) {
				int index = -1;
				auto minValue = get<0>(value);
				for (int i = 1; i < aNeighborhood.size(); ++i) {
					auto current = get<0>(aNeighborhood[i]);
					//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, get<1>(aNeighborhood[0]), get<1>(aNeighborhood[i]));
					if (current <= minValue && aNeighborhood.is_inside_valid_region(i)) {
						index = i;
						minValue = current;
					}
				}
				if (index != -1) {
					get<1>(value) = -1 * (1 + get_linear_access_index(gridView.dimensions(), aNeighborhood.view_index(index)));
				}
				aConvergenceState.signal();
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
	typedef Tuple<float, int32_t> Value2;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value2, tDimension>,
			VonNeumannNeighborhood<tDimension>,
			LocalMinimaConnectedComponentRule,
			LocalMinimaEquivalenceGlobalState<int32_t>> LocalMinimaAutomaton;
	auto localMinima = unaryOperatorOnLocator(deviceGradient, LocalMinimumLabel());

	LocalMinimaEquivalenceGlobalState<int32_t> globalState;

	thrust::device_vector<int32_t> buffer;
	buffer.resize(elementCount(deviceGradient) + 1);
	globalState.manager = EquivalenceManager<int32_t>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
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

template<int tDimension>
void distanceBasedWShed(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int32_t, float> Value;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
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

template<int tDimension, int tIterationLimit>
void distanceBasedWShedAsync(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int32_t, float> Value;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
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


template<int tDimension, typename TView>
void lowerCompletion(TView aInput)
{
	//lower completion
	device_image<float, tDimension> tmp(aInput.dimensions());
	transform_locator(aInput, view(tmp), HandlePlateauBorder());
	copy(const_view(tmp), aInput);
}

template<int tDimension>
void steepestDescentWShedSimple(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int32_t> Value;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedSteepestDescentRule,
			ConvergenceFlag> WatershedAutomaton;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));

	lowerCompletion<tDimension>(view(deviceGradient));
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), view(labels));
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()));
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension>(aInput.dimensions()));

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

template<int tDimension>
void steepestDescentWShedPointer(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int32_t> Value;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedByPointer,
			ConvergenceFlag> WatershedAutomaton;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));
	AggregatingTimerSet<2, int> timer;
	device_flag convergenceFlag;

	lowerCompletion<tDimension>(view(deviceGradient));
	initLocalMinimaLabels<tDimension>(const_view(deviceGradient), view(labels), timer, convergenceFlag);
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), view(labels));
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()));
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

	std::cout << timer.createReport({"Local minima search", "Wshed iterations"});
	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labels));
	copy(const_view(labels), aOutput);
}


template<int tDimension>
void steepestDescentWShedSimpleAsync(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int32_t> Value;
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
	device_image<int32_t, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));

	lowerCompletion<tDimension>(view(deviceGradient));
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), view(labels));
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()));
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension>(aInput.dimensions()));

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

template<int tDimension>
void steepestDescentWShedGlobalState(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int32_t> Value;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedSteepestDescentGlobalStateRule,
			LocalMinimaEquivalenceGlobalState<int32_t>> WatershedAutomaton;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));

	lowerCompletion<tDimension>(view(deviceGradient));
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), view(labels));
	//auto wshed = nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()));
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension>(aInput.dimensions()));

	device_flag convergenceFlag;
	LocalMinimaEquivalenceGlobalState<int32_t> globalState;

	thrust::device_vector<int32_t> buffer;
	buffer.resize(elementCount(aInput) + 1);
	globalState.manager = EquivalenceManager<int32_t>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
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

template<int tDimension>
void steepestDescentWShedAsyncGlobalState(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
}

template<int tDimension>
void steepestDescentWShedPointers(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{

}

//******************************************************************************

template<int tDimension>
void runWatershedTransformationDim(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	switch (aOptions.wshedVariant) {
	case WatershedVariant::DistanceBased:
		distanceBasedWShed<tDimension>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::DistanceBasedAsync:
		distanceBasedWShedAsync<tDimension, 10000>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::DistanceBasedAsyncLimited:
		distanceBasedWShedAsync<tDimension, 2>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentSimple:
		steepestDescentWShedSimple<tDimension>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentSimpleAsync:
		steepestDescentWShedSimpleAsync<tDimension>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentGlobalState:
		steepestDescentWShedGlobalState<tDimension>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentPointer:
		steepestDescentWShedPointer<tDimension>(aInput, aOutput, aOptions);
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
	runWatershedTransformationDim<3>(aInput, aOutput, aOptions);
}
