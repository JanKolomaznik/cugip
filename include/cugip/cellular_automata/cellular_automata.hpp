#pragma once

#include <cugip/image.hpp>
#include <cugip/for_each.hpp>

#include <cugip/equivalence_management.hpp>
#include <cugip/cellular_automata/neighborhood.hpp>
#include <cugip/cellular_automata/rules.hpp>

namespace cugip {


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



} //namespace cugip
