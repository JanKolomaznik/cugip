#pragma once

namespace cugip {

struct VonNeumannNeighborhood
{};

struct MooreNeighborhood
{};

struct Options
{};

template<typename TGrid, typename TNeighborhood, typename TRule, typename TOptions>
class CellularAutomata {
public:
	void
	intialize();

	void
	iterate(int aIterationCount);

	int
	run_until_equilibrium();
};


} //namespace cugip
