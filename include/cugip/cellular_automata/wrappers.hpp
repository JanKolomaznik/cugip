#pragma once

#include <cugip/image.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/cellular_automata/cellular_automata.hpp>
#include <cugip/cellular_automata/async_cellular_automata.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/view_arithmetics.hpp>

namespace cugip {

template<typename TView, typename TRule>
void runSimpleCellularAutomaton(
		TView aInput,
		TView aOutput,
		TRule aRule)
{
	typedef typename TView::value_type Value;
	constexpr int cDimension = dimension<TView>::value;

	typedef CellularAutomatonWithGlobalState<
			Grid<Value, cDimension>,
			MooreNeighborhood<cDimension>,
			TRule,
			ConvergenceFlag> Automaton;

	device_flag convergenceFlag;
	ConvergenceFlag convergenceGlobalState;
	convergenceGlobalState.mDeviceFlag = convergenceFlag.view();

	Automaton automaton;
	automaton.initialize(aInput, convergenceGlobalState);

	int iteration = 0;
	do {
		automaton.iterate(1);
	} while (!convergenceGlobalState.is_finished());

	copy(automaton.getCurrentState(), aOutput);
}

} // namespace cugip
