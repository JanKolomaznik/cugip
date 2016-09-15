#pragma once

#include <cugip/image.hpp>
#include <cugip/for_each.hpp>
#include <cugip/transform.hpp>

#include <cugip/equivalence_management.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/functors.hpp>
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


template<typename TNeighborhood, typename TRule>
struct CellOperation
{
	CellOperation(int aIteration, TRule aRule)
		: mRule(aRule)
		, mIteration(aIteration)
	{}

	template<typename TLocator>
	CUGIP_DECL_DEVICE typename TLocator::value_type
	operator()(TLocator aLocator) {
		typedef NeighborhoodAccessor<TLocator, TNeighborhood> Accessor;

		return mRule(mIteration, Accessor(aLocator, TNeighborhood()));
	}

	TRule mRule;
	int mIteration;
};

template<typename TNeighborhood, typename TRule, typename TGlobalState>
struct CellOperationWithGlobalState
{
	CellOperationWithGlobalState(int aIteration, TRule aRule, TGlobalState aGlobalState)
		: mRule(aRule)
		, mIteration(aIteration)
		, mGlobalState(aGlobalState)
	{}

	template<bool tUsesGlobalState, typename TDummy = int>
	struct CallWrapper;

	template<typename TDummy>
	struct CallWrapper<false, TDummy>
	{
		template<typename TAccessor>
		CUGIP_DECL_DEVICE
		static typename TAccessor::value_type
		invoke(
			int aIteration,
			TRule &aRule,
			TAccessor aAccessor,
			TGlobalState aGlobalState)
		{
			return aRule(aIteration, aAccessor);
			//aOutLocator.get() = aRule(aIteration, aAccessor);
			//aLocator2.get() = mRule(mIteration, Accessor(aLocator1, TNeighborhood()));
		}
	};

	template<typename TDummy>
	struct CallWrapper<true, TDummy>
	{
		template<typename TAccessor>
		CUGIP_DECL_DEVICE
		static typename TAccessor::value_type
		invoke(
			int aIteration,
			TRule &aRule,
			TAccessor aAccessor,
			TGlobalState aGlobalState)
		{
			return aRule(aIteration, aAccessor, aGlobalState);
			//aOutLocator.get() = mRule(mIteration, aAccessor);
			//aOutLocator.get() = aRule(aIteration, aAccessor, aGlobalState);
			//aLocator2.get() = mRule(mIteration, Accessor(aLocator1, TNeighborhood()));
		}
	};

	template<typename TLocator>
	CUGIP_DECL_DEVICE typename TLocator::value_type
	operator()(TLocator aLocator) {
		typedef NeighborhoodAccessor<TLocator, TNeighborhood> Accessor;
		typedef CallWrapper<is_callable<TRule, int, Accessor, TGlobalState>::value> Wrapper;
		return Wrapper::invoke(
				mIteration,
				mRule,
				Accessor(aLocator, TNeighborhood()),
				mGlobalState);
		//aLocator2.get() = mRule(mIteration,);
	}


	/*template<typename TLocator1, typename TLocator2>
	CUGIP_DECL_DEVICE void
	operator()(TLocator1 aLocator1, TLocator2 aLocator2) {
		typedef NeighborhoodAccessor<TLocator1, TNeighborhood> Accessor;

		CallWrapper<is_callable<TRule, int, Accessor, TGlobalState>::value>::invoke(mIteration, mRule, aLocator2, Accessor(aLocator1, TNeighborhood()), mGlobalState);
		//aLocator2.get() = mRule(mIteration,);
	}*/

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

/*
template<typename TGrid, typename TNeighborhood, typename TRule, typename TGlobalState = DummyGlobalState>
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
		CUGIP_DFORMAT("Cellular automaton initialization. Space: %1%", aView.dimensions());
		mIteration = 0;
		mImages[0].resize(aView.dimensions());
		mImages[1].resize(aView.dimensions());
		copy(aView, view(mImages[0]));

		mGlobalState.initialize();
		CUGIP_CHECK_ERROR_STATE("Cellular automaton initialization:");
	}

	void
	iterate(int aIterationCount)
	{
		CUGIP_CHECK_ERROR_STATE("Cellular automaton before iteration");
		for (int i = 0; i < aIterationCount; ++i) {
			CUGIP_DFORMAT("Cellular automaton iteration [%1%]", mIteration+1);
			transform_locator(const_view(mImages[mIteration % 2]), view(mImages[(mIteration + 1) % 2]), CellOperation<TNeighborhood, TRule, TGlobalState>(mIteration, mRule, mGlobalState));
			mGlobalState.postprocess(view(mImages[(mIteration + 1) % 2]));
			++mIteration;
			CUGIP_DFORMAT("Cellular automaton iteration [%1%] finished.", mIteration);
		}
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	typename State::const_view_t
	getCurrentState() const
	{
		CUGIP_DFORMAT("Cellular automaton getCurrentState. After iteration [%1%]", mIteration);
		return const_view(mImages[mIteration % 2]);
	}

	int
	run_until_equilibrium();

protected:
	int mIteration;
	std::array<State, 2> mImages;

	TRule mRule;
	TGlobalState mGlobalState;
};*/

template<typename TDerived, typename TRule, typename TState, int tBufferCount = 2>
class CellularAutomatonBaseCRTP {
public:
	typedef TState State;
	typedef TRule Rule;
	typedef typename State::const_view_t InputView;
	typedef typename State::view_t OutputView;

	template<typename TInputView>
	void
	initialize(Rule aRule, TInputView aView)
	{
		mRule = aRule;
		initialize(aView);
	}

	template<typename TInputView>
	void
	initialize(TInputView aView)
	{
		CUGIP_DFORMAT("Cellular automaton initialization. Space: %1%", aView.dimensions());
		mIteration = 0;
		for (auto &image : mImages) {
			image.resize(aView.dimensions());
		}
		copy(aView, view(mImages[0]));

		static_cast<TDerived *>(this)->postInitialization();
		//mGlobalState.initialize();
		CUGIP_CHECK_ERROR_STATE("Cellular automaton initialization:");
	}

	void
	iterate(int aIterationCount)
	{
		CUGIP_CHECK_ERROR_STATE("Cellular automaton before iteration");
		for (int i = 0; i < aIterationCount; ++i) {
			CUGIP_DFORMAT("Cellular automaton iteration [%1%]", mIteration+1);
			static_cast<TDerived *>(this)->computeNextIteration();
			/*transform_locator(const_view(mImages[mIteration % 2]), view(mImages[(mIteration + 1) % 2]), CellOperation<TNeighborhood, TRule, TGlobalState>(mIteration, mRule, mGlobalState));
			mGlobalState.postprocess(view(mImages[(mIteration + 1) % 2]));*/
			++mIteration;
			CUGIP_DFORMAT("Cellular automaton iteration [%1%] finished.", mIteration);
		}
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	typename State::const_view_t
	getCurrentState() const
	{
		CUGIP_DFORMAT("Cellular automaton getCurrentState. After iteration [%1%]", mIteration);
		return const_view(mImages[mIteration % tBufferCount]);
	}

	int
	run_until_equilibrium();

protected:
	InputView currentInputView()
	{
		return const_view(mImages[mIteration % tBufferCount]);
	}

	OutputView currentOutputView()
	{
		return view(mImages[(mIteration + 1) % tBufferCount]);
	}

	int mIteration;
	std::array<State, tBufferCount> mImages;

	Rule mRule;
};

template<typename TGrid, typename TNeighborhood, typename TRule>
class CellularAutomaton
	: public CellularAutomatonBaseCRTP<
			CellularAutomaton<TGrid, TNeighborhood, TRule>,
			TRule,
			device_image<typename TGrid::Element, TGrid::cDimension>>
{
public:
	typedef device_image<typename TGrid::Element, TGrid::cDimension> State;
	typedef CellularAutomatonBaseCRTP<CellularAutomaton<TGrid, TNeighborhood, TRule>, TRule, State> Predecessor;
	typedef TGrid Grid;
	typedef TRule Rule;
	friend Predecessor;

	using Predecessor::initialize;

protected:
	void postInitialization()
	{}

	void computeNextIteration()
	{
		transform_locator(
			this->currentInputView(),
			this->currentOutputView(),
			CellOperation<TNeighborhood, TRule>(this->mIteration, this->mRule));
	}
};

template<typename TGrid, typename TNeighborhood, typename TRule, typename TGlobalState = DummyGlobalState/*, typename TOptions*/>
class CellularAutomatonWithGlobalState
	: public CellularAutomatonBaseCRTP<
		CellularAutomatonWithGlobalState<TGrid, TNeighborhood, TRule, TGlobalState>,
		TRule,
		device_image<typename TGrid::Element, TGrid::cDimension>>
{
public:
	typedef device_image<typename TGrid::Element, TGrid::cDimension> State;
	typedef CellularAutomatonBaseCRTP<
			CellularAutomatonWithGlobalState<TGrid, TNeighborhood, TRule, TGlobalState>,
			TRule,
			State> Predecessor;
	typedef TGrid Grid;
	typedef TRule Rule;
	friend Predecessor;

	using Predecessor::initialize;

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

protected:
	void postInitialization()
	{
		mGlobalState.initialize();
	}

	void computeNextIteration()
	{
		mGlobalState.preprocess(this->currentInputView());
		transform_locator(
			this->currentInputView(),
			this->currentOutputView(),
			CellOperationWithGlobalState<TNeighborhood, TRule, TGlobalState>(this->mIteration, this->mRule, mGlobalState));
		mGlobalState.postprocess(this->currentOutputView());
	}

	TGlobalState mGlobalState;
};


template<typename TGrid, typename TNeighborhood, typename TRule, typename TGlobalState = DummyGlobalState/*, typename TOptions*/>
class AsyncCellularAutomatonWithGlobalState
	: public CellularAutomatonBaseCRTP<
		AsyncCellularAutomatonWithGlobalState<TGrid, TNeighborhood, TRule, TGlobalState>,
		TRule,
		device_image<typename TGrid::Element, TGrid::cDimension>,
		1>
{
public:
	typedef device_image<typename TGrid::Element, TGrid::cDimension> State;
	typedef CellularAutomatonBaseCRTP<
			AsyncCellularAutomatonWithGlobalState<TGrid, TNeighborhood, TRule, TGlobalState>,
			TRule,
			State,
			1> Predecessor;
	typedef TGrid Grid;
	typedef TRule Rule;
	friend Predecessor;

	using Predecessor::initialize;

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

protected:
	void postInitialization()
	{
		mGlobalState.initialize();
	}

	void computeNextIteration()
	{
		transform_locator(
			this->currentInputView(),
			this->currentOutputView(),
			CellOperationWithGlobalState<TNeighborhood, TRule, TGlobalState>(this->mIteration, this->mRule, mGlobalState));
		mGlobalState.postprocess(this->currentOutputView());
	}

	TGlobalState mGlobalState;
};

} //namespace cugip
