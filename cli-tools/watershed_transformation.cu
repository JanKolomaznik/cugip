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

using namespace cugip;


struct InitWatershed
{
	CUGIP_DECL_HYBRID
	Tuple<float, int64_t, float>
	operator()(float aGradient, int64_t aLocalMinimum) const
	{
		return Tuple<float, int64_t, float>(aGradient, aLocalMinimum, aLocalMinimum > 0 ? 0 : 1.0e15);
	}
};

struct ZipGradientAndLabel
{
	CUGIP_DECL_HYBRID
	Tuple<float, int64_t>
	operator()(float aGradient, int64_t aLocalMinimum) const
	{
		return Tuple<float, int64_t>(aGradient, aLocalMinimum);
	}
};


template<int tDimension>
void distanceBasedWShed(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int64_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	typedef Tuple<float, int64_t, float> Value;
	typedef Tuple<float, int64_t> Value2;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value2, tDimension>,
			VonNeumannNeighborhood<tDimension>,
			LocalMinimaConnectedComponentRule,
			LocalMinimaEquivalenceGlobalState<int64_t>> LocalMinimaAutomaton;
	typedef CellularAutomatonWithGlobalState<
			Grid<Value, tDimension>,
			MooreNeighborhood<tDimension>,
			WatershedRule,
			ConvergenceFlag> WatershedAutomaton;
	device_image<float, tDimension> deviceGradient(aInput.dimensions());
	copy(aInput, view(deviceGradient));
	auto localMinima = unaryOperatorOnLocator(const_view(deviceGradient), LocalMinimumLabel());

	LocalMinimaEquivalenceGlobalState<int64_t> globalState;

	device_flag convergenceFlag;
	thrust::device_vector<int64_t> buffer;
	buffer.resize(elementCount(aInput) + 1);
	globalState.manager = EquivalenceManager<int64_t>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	LocalMinimaAutomaton localMinimumAutomaton;
	localMinimumAutomaton.initialize(
		nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), localMinima),
		globalState);
	localMinimumAutomaton.iterate(500);

	/*auto wshed = nAryOperator(InitWatershed(), const_view(deviceGradient), getDimension(localMinimumAutomaton.getCurrentState(), IntValue<1>()));

	ConvergenceFlag convergenceGlobalState;
	convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

	WatershedAutomaton automaton;
	automaton.initialize(wshed, convergenceGlobalState);

	do {
		automaton.iterate();
	} while (!convergenceGlobalState.is_finished());

	auto state = automaton.getCurrentState();
	device_image<int64_t, tDimension> tmpState(state.dimensions());
	copy(getDimension(state, IntValue<1>()), view(tmpState));
	copy(const_view(tmpState), aOutput);*/
}



template<int tDimension>
void runWatershedTransformationDim(
		const_host_image_view<const float, tDimension> aInput,
		host_image_view<int64_t, tDimension> aOutput,
		const WatershedOptions &aOptions)
{
	distanceBasedWShed<tDimension>(aInput, aOutput, aOptions);
}

void runWatershedTransformation(
		const_host_image_view<const float, 2> aInput,
		host_image_view<int64_t, 2> aOutput,
		const WatershedOptions &aOptions)
{
	runWatershedTransformationDim<2>(aInput, aOutput, aOptions);
}

void runWatershedTransformation(
		const_host_image_view<const float, 3> aInput,
		host_image_view<int64_t, 3> aOutput,
		const WatershedOptions &aOptions)
{
	runWatershedTransformationDim<3>(aInput, aOutput, aOptions);
}
