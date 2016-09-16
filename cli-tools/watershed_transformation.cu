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


template<int tDimension>
void distanceBasedWShed(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int32_t, float> Value;
	typedef Tuple<float, int32_t> Value2;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value2, tDimension>,
			VonNeumannNeighborhood<tDimension>,
			LocalMinimaConnectedComponentRule,
			LocalMinimaEquivalenceGlobalState<int32_t>> LocalMinimaAutomaton;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedRule,
			ConvergenceFlag> WatershedAutomaton;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	device_image<int32_t, tDimension> labels(aInput.dimensions());
	copy(aInput, view(deviceGradient));
	device_flag convergenceFlag;
	AggregatingTimerSet<2, int> timer;
	{
		auto localMinima = unaryOperatorOnLocator(const_view(deviceGradient), LocalMinimumLabel());

		LocalMinimaEquivalenceGlobalState<int32_t> globalState;

		thrust::device_vector<int32_t> buffer;
		buffer.resize(elementCount(aInput) + 1);
		globalState.manager = EquivalenceManager<int32_t>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
		globalState.mDeviceFlag = convergenceFlag.view();
		globalState.manager.initialize();

		LocalMinimaAutomaton localMinimumAutomaton;
		localMinimumAutomaton.initialize(
			nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), localMinima),
			globalState);

		int iteration = 0;
		do {
			auto interval = timer.start(0, iteration++);
			localMinimumAutomaton.iterate(1);
		} while (!globalState.is_finished());
		//localMinimumAutomaton.iterate(100);
		copy(getDimension(localMinimumAutomaton.getCurrentState(), IntValue<1>()), view(labels));
	}

	auto wshed = nAryOperator(InitWatershed(), const_view(deviceGradient), const_view(labels));

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
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()));

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
void steepestDescentWShedAsyncSimple(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int32_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{

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
	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<tDimension, int32_t>(aInput.dimensions()));

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
	case WatershedVariant::SteepestDescentSimple:
		steepestDescentWShedSimple<tDimension>(aInput, aOutput, aOptions);
		break;
	case WatershedVariant::SteepestDescentGlobalState:
		steepestDescentWShedGlobalState<tDimension>(aInput, aOutput, aOptions);
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
	runWatershedTransformationDim<2>(aInput, aOutput, aOptions);
}

void runWatershedTransformation(
		const_host_image_view<const float, 3> aInput,
		host_image_view<int32_t, 3> aOutput,
		const WatershedOptions &aOptions)
{
	runWatershedTransformationDim<3>(aInput, aOutput, aOptions);
}
