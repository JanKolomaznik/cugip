#pragma once

#include <cugip/image.hpp>
#include <cugip/for_each.hpp>
#include <cugip/transform.hpp>

#include <cugip/equivalence_management.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/functors.hpp>
#include <cugip/cellular_automata/rules.hpp>
#include <cugip/cellular_automata/cellular_automata.hpp>

namespace cugip {

/** \addtogroup math
 * @{
 **/
template<typename TNeighborhood, typename TRule, typename TGlobalState>
struct AsyncCellOperationWithGlobalState
{
	AsyncCellOperationWithGlobalState(int aIteration, TRule aRule, TGlobalState aGlobalState)
		: mRule(aRule)
		, mIteration(aIteration)
		, mGlobalState(aGlobalState)
	{}

	template<typename TLocator, typename TSemiGLobalState>
	CUGIP_DECL_DEVICE typename TLocator::value_type
	operator()(TLocator aLocator, TSemiGLobalState aSemiGlobalState)
	{
		typedef NeighborhoodAccessor<TLocator, TNeighborhood> Accessor;

		return mRule(mIteration, Accessor(aLocator, TNeighborhood()), aSemiGlobalState);
	}

	template<typename TLocator, typename TSemiGLobalState>
	CUGIP_DECL_DEVICE typename TLocator::value_type
	update_global(TLocator aLocator, TSemiGLobalState aSemiGlobalState)
	{
		typedef NeighborhoodAccessor<TLocator, TNeighborhood> Accessor;
		return mRule(mIteration, Accessor(aLocator, TNeighborhood()), mGlobalState);
	}

	template<typename TSemiGLobalState>
	CUGIP_DECL_DEVICE void
	update_global2(TSemiGLobalState aSemiGlobalState)
	{
		aSemiGlobalState.update_global(mGlobalState);
		//typedef NeighborhoodAccessor<TLocator, TNeighborhood> Accessor;
		//return mRule(mIteration, Accessor(aLocator, TNeighborhood()), mGlobalState);
	}

	TRule mRule;
	int mIteration;
	TGlobalState mGlobalState;
};


template <typename TInView, typename TOutView, typename TRule, typename TPolicy, typename TSemiGlobalState>
CUGIP_GLOBAL void
kernel_hidden_updates(TInView aInView, TOutView aOutView, TRule aRule, TPolicy aPolicy)
{
	typename TInView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView>::value>();
	typename TInView::extents_t extents = aInView.dimensions();
	typedef typename TPolicy::RegionSize Size;
	__shared__ cugip::detail::SharedMemory<typename TInView::value_type, Size> buffer;

	__shared__ TSemiGlobalState semiGlobalState;
	__syncthreads();
	semiGlobalState.initialize();

	auto loadedRegion = aPolicy.regionForBlock();
	auto corner = mapBlockIdxToViewCoordinates<dimension<TInView>::value>() + loadedRegion.corner;
	auto preloadCoords = coord - corner;// current element coords in the preload buffer

	auto dataView = makeDeviceImageView(&(buffer.get(Int3())), to_vector(Size()));
	typedef decltype(dataView) DataView;
	buffer.load(aInView, corner);
	//buffer.loadZeroOut(aInView, corner);
	__syncthreads();

	/*if (is_in_block(0,1,0) && is_in_thread(0,0,0)) {
		printf("%d, %d, %d\n", preloadCoords[0], preloadCoords[1], preloadCoords[2]);
	}*/
	typedef image_locator<DataView, BorderHandlingTraits<border_handling_enum::NONE>> Locator;
	Locator itemLocator(dataView, preloadCoords);
	do {
		semiGlobalState.preprocess();
		__syncthreads();
		auto newState = itemLocator.get();
		if (coord < extents) {
			newState = aRule(itemLocator, semiGlobalState.view());
			//newState = aRule(itemLocator, aRule.mGlobalState);
		}
		__syncthreads();

		buffer.get(preloadCoords) = newState;

		__syncthreads();
	} while (!semiGlobalState.is_finished());

	aRule.update_global2(semiGlobalState.view());
	__syncthreads();
	if (coord < extents) {
		//buffer.get(preloadCoords) = aRule.update_global(itemLocator, semiGlobalState.view());
		aOutView[coord] = buffer.get(preloadCoords);
	}
}

template <typename TInView, typename TOutView, typename TRule, typename TSemiGlobalState>
void hidden_updates(TInView aInView, TOutView aOutView, TRule aRule, cudaStream_t aCudaStream = 0) {
	typedef PreloadingTransformLocatorPolicy<TInView, 1> Policy;
	Policy aPolicy;

	// TODO - do this only in code processed by nvcc
	dim3 blockSize = aPolicy.blockSize();
	dim3 gridSize = aPolicy.gridSize(aInView);

	kernel_hidden_updates<TInView, TOutView, TRule, Policy, TSemiGlobalState>
		<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aRule, aPolicy);
}


template<typename TGrid, typename TNeighborhood, typename TRule, typename TGlobalState, typename TSemiGlobalState/*, typename TOptions*/>
class AsyncCellularAutomatonWithGlobalState
	: public CellularAutomatonBaseCRTP<
		AsyncCellularAutomatonWithGlobalState<TGrid, TNeighborhood, TRule, TGlobalState, TSemiGlobalState>,
		TRule,
		device_image<typename TGrid::Element, TGrid::cDimension>,
		2>
{
public:
	typedef device_image<typename TGrid::Element, TGrid::cDimension> State;
	typedef CellularAutomatonBaseCRTP<
			AsyncCellularAutomatonWithGlobalState<TGrid, TNeighborhood, TRule, TGlobalState, TSemiGlobalState>,
			TRule,
			State,
			2> Predecessor;
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
		typedef AsyncCellOperationWithGlobalState<
				TNeighborhood,
				TRule,
				TGlobalState> RuleWrapper;
		mGlobalState.preprocess(this->currentInputView());
		hidden_updates<
			decltype(this->currentInputView()),
			decltype(this->currentOutputView()),
			RuleWrapper,
			TSemiGlobalState>(
				this->currentInputView(),
				this->currentOutputView(),
				RuleWrapper(this->mIteration, this->mRule, mGlobalState));
		mGlobalState.postprocess(this->currentOutputView());
	}

	TGlobalState mGlobalState;
};

template<typename TGrid, typename TNeighborhood, typename TRule, typename TGlobalState, typename TSemiGlobalState/*, typename TOptions*/>
class AsyncCellularAutomatonWithGlobalStateAndPointer
	: public CellularAutomatonBaseCRTP<
		AsyncCellularAutomatonWithGlobalStateAndPointer<TGrid, TNeighborhood, TRule, TGlobalState, TSemiGlobalState>,
		TRule,
		device_image<typename TGrid::Element, TGrid::cDimension>,
		2>
{
public:
	typedef device_image<typename TGrid::Element, TGrid::cDimension> State;
	typedef CellularAutomatonBaseCRTP<
			AsyncCellularAutomatonWithGlobalStateAndPointer<TGrid, TNeighborhood, TRule, TGlobalState, TSemiGlobalState>,
			TRule,
			State,
			2> Predecessor;
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
		typedef AsyncCellOperationWithGlobalState<
				TNeighborhood,
				TRule,
				TGlobalState> RuleWrapper;
		mGlobalState.preprocess(this->currentInputView());
		hidden_updates<
			decltype(this->currentInputView()),
			decltype(this->currentOutputView()),
			RuleWrapper,
			TSemiGlobalState>(
				this->currentInputView(),
				this->currentOutputView(),
				RuleWrapper(this->mIteration, this->mRule, mGlobalState));
		mGlobalState.postprocess(this->currentOutputView());
	}

	TGlobalState mGlobalState;
};

/**
 * @}
 **/

} //namespace cugip
