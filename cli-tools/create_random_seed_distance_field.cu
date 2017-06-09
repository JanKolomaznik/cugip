
#include <cugip/cellular_automata/wrappers.hpp>

struct DistanceRule
{
	template<typename TNeighborhood, typename TConvergenceFlag>
	CUGIP_DECL_DEVICE
	float operator()(int aIteration, TNeighborhood aNeighborhood, TConvergenceFlag aConvergenceState) const
	{
		//input, label, distance
		auto value = aNeighborhood[0];
		auto minValue = aNeighborhood[1] + magnitude(aNeighborhood.offset(1));
		for (int i = 2; i < aNeighborhood.size(); ++i) {
			auto distance = aNeighborhood[i] + magnitude(aNeighborhood.offset(i));
			if (distance < minValue) {
				minValue = distance;
			}
		}
		if (minValue < value) {
			value = minValue;
			aConvergenceState.signal();
		}
		return value;
	}
};

void computeDistanceField(cugip::host_image_view<float, 3> field)
{
	runSimpleCellularAutomaton(field, field, DistanceRule());
}
