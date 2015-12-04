#include <cugip/cellular_automata/cellular_automata.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/copy.hpp>
#include <cugip/tuple.hpp>
#include <thrust/device_vector.h>

#include "watershed_transformation.hpp"

#include <boost/timer/timer.hpp>

struct InitWatershed
{
	CUGIP_DECL_HYBRID
	cugip::Tuple<float, int, float>
	operator()(int aGradient, int aLocalMinimum) const
	{
		return cugip::Tuple<float, int, float>(aGradient, aLocalMinimum, aLocalMinimum > 0 ? 0 : 999999);
	}
};

struct ZipGradientAndLabel
{
	CUGIP_DECL_HYBRID
	cugip::Tuple<float, int>
	operator()(float aGradient, int aLocalMinimum) const
	{
		return cugip::Tuple<float, int>(aGradient, aLocalMinimum);
	}
};

void
watershedTransformation(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::host_image_view<int, 3> aLabels,
	Options aOptions)
{
	using namespace cugip;
	typedef Tuple<float, int, float> Value;
	typedef Tuple<float, int> Value2;

	device_image<float, 3> data(aData.dimensions());
	device_image<int, 3> labelImage(aData.dimensions());
	copy(aData, view(data));
	device_flag convergenceFlag;

	{
		CellularAutomaton<Grid<Value2, 3>, VonNeumannNeighborhood<3>, LocalMinimaConnectedComponentRule, LocalMinimaEquivalenceGlobalState> localMinimumAutomaton;

		thrust::device_vector<int> buffer(elementCount(aData) + 1);
		auto localMinima = unaryOperatorOnLocator(const_view(data), LocalMinimumLabel());
		/*copy(localMinima, view(labelImage));
		copy(view(labelImage), aLabels);
		return;*/
		LocalMinimaEquivalenceGlobalState globalState;
		globalState.manager = EquivalenceManager<int>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
		globalState.mDeviceFlag = convergenceFlag.view();
		localMinimumAutomaton.initialize(nAryOperator(ZipGradientAndLabel(), const_view(data), localMinima), globalState);
		do {
			convergenceFlag.reset_host();
			localMinimumAutomaton.iterate(1);
		} while (convergenceFlag.check_host());
		copy(getDimension(localMinimumAutomaton.getCurrentState(), IntValue<1>()), view(labelImage));
	}

	CellularAutomaton<Grid<Value, 3>, MooreNeighborhood<3>, WatershedRule, ConvergenceFlag> automaton;
	ConvergenceFlag convergenceGlobalState{ convergenceFlag.view() };

	auto wshed = nAryOperator(InitWatershed(), const_view(data), const_view(labelImage));
	automaton.initialize(wshed, convergenceGlobalState);
	do {
		convergenceFlag.reset_host();
		automaton.iterate(1);
	} while (convergenceFlag.check_host());

	auto state = automaton.getCurrentState();
	copy(getDimension(state, IntValue<1>()), view(labelImage));
	copy(const_view(labelImage), aLabels);
}
