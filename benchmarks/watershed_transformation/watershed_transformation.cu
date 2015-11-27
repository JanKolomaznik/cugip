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
	copy(aData, view(data));
	CellularAutomaton<Grid<Value2, 3>, VonNeumannNeighborhood<3>, LocalMinimaConnectedComponentRule, LocalMinimaEquivalenceGlobalState> localMinimumAutomaton;
	CellularAutomaton<Grid<Value, 3>, MooreNeighborhood<3>, WatershedRule> automaton;

	thrust::device_vector<int> buffer(elementCount(aData) + 1);
	auto localMinima = unaryOperatorOnLocator(const_view(data), LocalMinimumLabel());

	LocalMinimaEquivalenceGlobalState globalState;
	globalState.manager = EquivalenceManager<int>(thrust::raw_pointer_cast(&buffer[0]), buffer.size());
	localMinimumAutomaton.initialize(nAryOperator(ZipGradientAndLabel(), const_view(data), localMinima), globalState);
	localMinimumAutomaton.iterate(50);

	auto wshed = nAryOperator(InitWatershed(), const_view(data), getDimension(localMinimumAutomaton.getCurrentState(), IntValue<1>()));
	automaton.initialize(wshed);
}
