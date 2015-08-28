#pragma once

#include <cugip/image.hpp>
#include <cugip/for_each.hpp>

#include <cugip/equivalence_management.hpp>

namespace cugip {

template<int tDimension>
struct VonNeumannNeighborhood;

template<>
struct VonNeumannNeighborhood<2>
{
	CUGIP_DECL_HYBRID int
	size() const
	{
		return 5;
	}

	CUGIP_DECL_HYBRID Int2
	offset(int aIndex) const
	{
		switch (aIndex) {
		case 0: return Int2(0, 0);
		case 1: return Int2(-1, 0);
		case 2: return Int2(1, 0);
		case 3: return Int2(0, -1);
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
	CUGIP_DECL_HYBRID int
	size() const
	{
		return 7;
	}

	CUGIP_DECL_HYBRID Int3
	offset(int aIndex) const
	{
		switch (aIndex) {
		case 0: return Int3(0, 0, 0);
		case 1: return Int3(-1, 0, 0);
		case 2: return Int3(1, 0, 0);
		case 3: return Int3(0, -1, 0);
		case 4: return Int3(0, 1, 0);
		case 5: return Int3(0, 0, -1);
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
	CUGIP_DECL_HYBRID int
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
};

template<>
struct MooreNeighborhood<3>
{
	CUGIP_DECL_HYBRID int
	size() const
	{
		return 27;
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
		case 9: return Int3(1, 1, 1);

		case 10: return Int3(-1, -1, 0);
		case 11: return Int3(0, -1, 0);
		case 12: return Int3(1, -1, 0);
		case 13: return Int3(-1, 0, 0);
		case 14: return Int3(1, 0, 0);
		case 15: return Int3(-1, 1, 0);
		case 16: return Int3(0, 1, 0);
		case 17: return Int3(1, 1, 0);

		case 18: return Int3(-1, -1, -1);
		case 19: return Int3(0, -1, -1);
		case 20: return Int3(1, -1, -1);
		case 21: return Int3(-1, 0, -1);
		case 22: return Int3(0, 0, -1);
		case 23: return Int3(1, 0, -1);
		case 24: return Int3(-1, 1, -1);
		case 25: return Int3(0, 1, -1);
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
	size() const
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

struct Options
{};

template < typename PotentiallyCallable, typename... Args>
struct is_callable
{
	typedef char (&no)  [1];
	typedef char (&yes) [2];

	template < typename T > struct dummy;

	template < typename CheckType>
	static yes check(dummy<decltype(std::declval<CheckType>()(std::declval<Args>()...))> *);
	template < typename CheckType>
	static no check(...);

	enum { value = sizeof(check<PotentiallyCallable>(0)) == sizeof(yes) };
};

template<typename TNeighborhood, typename TRule, typename TGlobalState>
struct CellOperation
{
	CellOperation(int aIteration, TRule aRule, TGlobalState aGlobalState)
		: mRule(aRule)
		, mIteration(aIteration)
		, mGlobalState(aGlobalState)
	{}

	template<bool tUsesGlobalState, typename TDummy = int>
	struct CallWrapper;
	
	template<typename TDummy>
	struct CallWrapper<false, TDummy>
	{
		template<typename TOutLocator, typename TAccessor>
		CUGIP_DECL_DEVICE
		static void invoke(int aIteration, TRule &aRule, TOutLocator aOutLocator, TAccessor aAccessor, TGlobalState aGlobalState)
		{
			aOutLocator.get() = aRule(aIteration, aAccessor);
			//aLocator2.get() = mRule(mIteration, Accessor(aLocator1, TNeighborhood()));
		}
	};

	template<typename TDummy>
	struct CallWrapper<true, TDummy>
	{
		template<typename TOutLocator, typename TAccessor>
		CUGIP_DECL_DEVICE
		static void invoke(int aIteration, TRule &aRule, TOutLocator aOutLocator, TAccessor aAccessor, TGlobalState aGlobalState)
		{
			//aOutLocator.get() = mRule(mIteration, aAccessor);
			aOutLocator.get() = aRule(aIteration, aAccessor, aGlobalState);
			//aLocator2.get() = mRule(mIteration, Accessor(aLocator1, TNeighborhood()));
		}
	};


	template<typename TLocator1, typename TLocator2>
	CUGIP_DECL_DEVICE void
	operator()(TLocator1 aLocator1, TLocator2 aLocator2) {
		typedef NeighborhoodAccessor<TLocator1, TNeighborhood> Accessor;

		CallWrapper<is_callable<TRule, int, Accessor, TGlobalState>::value>::invoke(mIteration, mRule, aLocator2, Accessor(aLocator1, TNeighborhood()), mGlobalState);
		//aLocator2.get() = mRule(mIteration,);
	}

	TRule mRule;
	int mIteration;
	TGlobalState mGlobalState;
};

template<typename TElement, int tDim>
struct Grid
{
	static const int cDimension = tDim;
	typedef TElement Element;
};

struct DummyGlobalState
{
	void
	initialize(){}

	template<typename TView>
	void
	postprocess(TView /*aView*/){}
};


template<typename TGrid, typename TNeighborhood, typename TRule, typename TGlobalState = DummyGlobalState/*, typename TOptions*/>
class CellularAutomaton {
public:
	typedef device_image<typename TGrid::Element, TGrid::cDimension> State;

	template<typename TInputView>
	void
	initialize(TRule aRule, TInputView aView, TGlobalState aGlobalState)
	{
		mGlobalState = aGlobalState;
		initialize(aRule, aView);
	}

	template<typename TInputView>
	void
	initialize(TInputView aView, TGlobalState aGlobalState)
	{
		mGlobalState = aGlobalState;
		initialize(aView);
	}

	template<typename TInputView>
	void
	initialize(TRule aRule, TInputView aView)
	{
		mRule = aRule;
		initialize(aView);
	}

	template<typename TInputView>
	void
	initialize(TInputView aView)
	{
		mIteration = 0;
		mImages[0].resize(aView.dimensions());
		mImages[1].resize(aView.dimensions());
		copy(aView, view(mImages[0]));

		mGlobalState.initialize();
	}

	void
	iterate(int aIterationCount)
	{
		for (int i = 0; i < aIterationCount; ++i) {
			for_each_locator(const_view(mImages[mIteration % 2]), view(mImages[(mIteration + 1) % 2]), CellOperation<TNeighborhood, TRule, TGlobalState>(mIteration, mRule, mGlobalState));
			mGlobalState.postprocess(view(mImages[(mIteration + 1) % 2]));
			++mIteration;
		}
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	typename State::const_view_t
	getCurrentState() const
	{
		return const_view(mImages[mIteration % 2]);
	}

	int
	run_until_equilibrium();

protected:
	int mIteration;
	std::array<State, 2> mImages;

	TRule mRule;
	TGlobalState mGlobalState;
};


struct ConwayRule
{
	template<typename TNeighborhood>
	CUGIP_DECL_HYBRID uint8_t
	operator()(int aIteration, TNeighborhood aNeighborhood) const
	{
		int sum = 0;
		for (int i = 1; i < aNeighborhood.size(); ++i) {
			sum += aNeighborhood[i];
		}
		if (aNeighborhood[0]) {
			if (sum < 2 || sum > 3) {
				return 0;
			}
			return 1;
		} else {
			if (sum == 3) {
				return 1;
			}
			return 0;
		}
	}
};

struct ConnectedComponentLabelingRule
{
	template<typename TNeighborhood>
	CUGIP_DECL_HYBRID int
	operator()(int aIteration, TNeighborhood aNeighborhood) const
	{
		auto value = aNeighborhood[0];
		if (value) {
			for (int i = 1; i < aNeighborhood.size(); ++i) {
				//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
				if (aNeighborhood[i] > 0 && aNeighborhood[i] < value) {
					value = aNeighborhood[i];
				}
			}
		}
		return value;
	}
};


struct EquivalenceGlobalState
{
	void
	initialize(){
		manager.initialize();
	}

	template<typename TView>
	void
	postprocess()
	{
		manager.compaction();
	}

	EquivalenceManager<int> manager;
};

struct ConnectedComponentLabelingRule2
{
	template<typename TNeighborhood>
	CUGIP_DECL_DEVICE int
	operator()(int aIteration, TNeighborhood aNeighborhood, EquivalenceGlobalState &aEquivalence) const
	{
		auto value = aNeighborhood[0];
		auto minValue = value;
		if (value) {
			for (int i = 1; i < aNeighborhood.size(); ++i) {
				//printf("%d %d - %d val = %d -> %d\n", threadIdx.x, threadIdx.y, i, aNeighborhood[0], aNeighborhood[i]);
				if (aNeighborhood[i] > 0 && aNeighborhood[i] < minValue) {
					minValue = aNeighborhood[i];
				}
			}
			if (minValue < value) {
				aEquivalence.manager.merge(minValue, value);
			}
		}
		return value;
	}
};

} //namespace cugip
