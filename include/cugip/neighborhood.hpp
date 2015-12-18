#pragma once

#include <cugip/math.hpp>

namespace cugip {

template<int tDimension>
struct VonNeumannNeighborhood;

template<>
struct VonNeumannNeighborhood<2>
{
	CUGIP_DECL_HYBRID constexpr int
	size() const
	{
		return 5;
	}

	CUGIP_DECL_HYBRID static constexpr int
	helperOffset(int aIndex) {
		// 0; -1; 0; 1
		return aIndex < 0 || aIndex > 3 ? 0 : (-1 * ((aIndex % 2) * (1 - aIndex / 2)) + (aIndex / 2 * (aIndex % 2)));
	}

	CUGIP_DECL_HYBRID constexpr Int2
	offset(int aIndex) const
	{
		return Int2(
			helperOffset(aIndex),
			helperOffset(aIndex - 1)
			);
	}

	CUGIP_DECL_HYBRID Int2
	offset2(int aIndex) const
	{
		switch (aIndex) {
		case 0: return Int2(0, 0);
		case 1: return Int2(-1, 0);
		case 2: return Int2(0, -1);
		case 3: return Int2(1, 0);
		case 4: return Int2(0, 1);
		default: CUGIP_ASSERT(false);
			break;
		}
		return Int2();
	}
};

template<>
struct VonNeumannNeighborhood<3>
{
	CUGIP_DECL_HYBRID constexpr int
	size() const
	{
		return 7;
	}

	CUGIP_DECL_HYBRID static constexpr int
	helperOffset(int aIndex) {
		// 0; 0; -1; 0; 0; 1
		return aIndex < 0 || aIndex > 5 ? 0 : (-1 * ((aIndex % 3 - aIndex % 2) * (1 - aIndex / 3)) / 2 + (aIndex / 3 *(aIndex % 3 - (aIndex - 3) % 2)) / 2);
	}

	CUGIP_DECL_HYBRID constexpr Int3
	offset(int aIndex) const
	{
		return Int3(
			helperOffset(aIndex + 1),
			helperOffset(aIndex),
			helperOffset(aIndex - 1)
			);
	}
	CUGIP_DECL_HYBRID Int3
	offset2(int aIndex) const
	{
		switch (aIndex) {
		case 0: return Int3(0, 0, 0);
		case 1: return Int3(-1, 0, 0);
		case 2: return Int3(0, -1, 0);
		case 3: return Int3(0, 0, -1);
		case 4: return Int3(1, 0, 0);
		case 5: return Int3(0, 1, 0);
		case 6: return Int3(0, 0, 1);
		default: CUGIP_ASSERT(false);
			break;
		}
		return Int3();
	}
};

template<int tDimension>
struct MooreNeighborhood;

template<>
struct MooreNeighborhood<2>
{
	CUGIP_DECL_HYBRID constexpr int
	size() const
	{
		return 9;
	}

	CUGIP_DECL_HYBRID Int2
	offset(int aIndex) const
	{


		switch (aIndex) {
		case 0: return Int2(0, 0);
		case 1: return Int2(-1, -1);
		case 2: return Int2(0, -1);
		case 3: return Int2(1, -1);
		case 4: return Int2(-1, 0);
		case 5: return Int2(1, 0);
		case 6: return Int2(-1, 1);
		case 7: return Int2(0, 1);
		case 8: return Int2(1, 1);
		default: CUGIP_ASSERT(false);
			break;
		}
		return Int2();
	}

	CUGIP_DECL_HYBRID Int2
	offset2(int aIndex) const
	{
		switch (aIndex) {
		case 0: return Int2(0, 0);
		case 1: return Int2(-1, -1);
		case 2: return Int2(0, -1);
		case 3: return Int2(1, -1);
		case 4: return Int2(-1, 0);
		case 5: return Int2(1, 0);
		case 6: return Int2(-1, 1);
		case 7: return Int2(0, 1);
		case 8: return Int2(1, 1);
		default: CUGIP_ASSERT(false);
			break;
		}
		return Int2();
	}
};

template<>
struct MooreNeighborhood<3>
{
	CUGIP_DECL_HYBRID constexpr int
	size() const
	{
		return 27;
	}

	CUGIP_DECL_HYBRID static constexpr int
	helperOffsetZ(int aIndex)
	{
		// -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 0
		return aIndex < 0 ? 0 : (aIndex / 9 - 1);
	}

	CUGIP_DECL_HYBRID static constexpr int
	helperOffsetY(int aIndex)
	{
		// -1 -1 -1 0 0 0 1 1 1 -1 -1 -1 0
		return aIndex < 0 ? 0 : ((aIndex / 3) % 3 - 1);
	}

	CUGIP_DECL_HYBRID static constexpr int
	helperOffsetX(int aIndex)
	{
		// -1 0 1 -1 0 1 -1 0 1 -1 0 1 -1
		return aIndex < 0 ? 0 : ((aIndex % 3) - 1);
	}

	template<int tMultiplier>
	CUGIP_DECL_HYBRID static constexpr Int3
	hemisphereOffset(int aIndex)
	{
		return Int3(
			tMultiplier * helperOffsetX(aIndex - 1),
			tMultiplier * helperOffsetY(aIndex - 1),
			tMultiplier * helperOffsetZ(aIndex - 1)
			);
	}

	CUGIP_DECL_HYBRID constexpr Int3
	offset2(int aIndex) const
	{
		return aIndex < 14 ? hemisphereOffset<1>(aIndex) : hemisphereOffset<-1>(27 - aIndex);
	}

	CUGIP_DECL_HYBRID Int3
	offset(int aIndex) const
	{
		switch (aIndex) {
		case 0: return Int3(0, 0, 0);
		case 1: return Int3(-1, -1, -1);
		case 2: return Int3(0, -1, -1);
		case 3: return Int3(1, -1, -1);
		case 4: return Int3(-1, 0, -1);
		case 5: return Int3(0, 0, -1);
		case 6: return Int3(1, 0, -1);
		case 7: return Int3(-1, 1, -1);
		case 8: return Int3(0, 1, -1);
		case 9: return Int3(1, 1, -1);

		case 10: return Int3(-1, -1, 0);
		case 11: return Int3(0, -1, 0);
		case 12: return Int3(1, -1, 0);
		case 13: return Int3(-1, 0, 0);

		case 14: return Int3(1, 0, 0);
		case 15: return Int3(-1, 1, 0);
		case 16: return Int3(0, 1, 0);
		case 17: return Int3(1, 1, 0);

		case 18: return Int3(-1, -1, 1);
		case 19: return Int3(0, -1, 1);
		case 20: return Int3(1, -1, 1);
		case 21: return Int3(-1, 0, 1);
		case 22: return Int3(0, 0, 1);
		case 23: return Int3(1, 0, 1);
		case 24: return Int3(-1, 1, 1);
		case 25: return Int3(0, 1, 1);
		case 26: return Int3(1, 1, 1);
		default: CUGIP_ASSERT(false);
			break;
		}
		return Int3();
	}
};

template<typename TLocator, typename TNeighborhood>
struct NeighborhoodAccessor
{
	CUGIP_DECL_HYBRID
	NeighborhoodAccessor(TLocator aLocator, TNeighborhood aNeighborhood)
		: mLocator(aLocator)
		, mNeighborhood(aNeighborhood)
	{}

	CUGIP_DECL_HYBRID int
	constexpr size() const
	{
		return mNeighborhood.size();
	}

	CUGIP_DECL_HYBRID typename TLocator::accessed_type
	operator[](int aIndex)
	{
		return mLocator[mNeighborhood.offset(aIndex)];
	}

	TLocator mLocator;
	TNeighborhood mNeighborhood;
};

} // namespace cugip
