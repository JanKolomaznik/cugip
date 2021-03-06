
#include "AutomatonWrapper.hpp"

#include <cugip/cellular_automata/cellular_automata.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/view_arithmetics.hpp>
#include <cugip/host_image.hpp>
#include <cugip/copy.hpp>
#include <cugip/tuple.hpp>
#include <cugip/reduce.hpp>
#include <thrust/device_vector.h>

using namespace cugip;

struct ColorToCell {
	CUGIP_DECL_HYBRID uint8_t
	operator()(element_rgb8_t aValue) const {
		int sum = aValue.data[0] + aValue.data[1] + aValue.data[2];
		if (sum > 300) {
			return 0;
		}
		return 1;
	}
};

struct CellToColor {
	CUGIP_DECL_HYBRID element_rgb8_t
	operator()(uint8_t aValue) const {
		if (aValue >0) {
			return element_rgb8_t{ 0, 0, 0 };
		}
		return element_rgb8_t{ 255, 255, 255 };
	}
};

struct ColorToLabel {
	ColorToLabel(Int2 aSize)
		: mSize(aSize)
	{}

	CUGIP_DECL_HYBRID int
	operator()(element_rgb8_t aValue, Int2 aCoords) const {
		int sum = aValue.data[0] + aValue.data[1] + aValue.data[2];
		if (sum > 300) {
			return 0;
		}
		return get_linear_access_index(mSize, aCoords);
	}
	Int2 mSize;
};


class AutomatonWrapper: public AAutomatonWrapper
{

	void
	setStartImage(const unsigned char *aBuffer, int aWidth, int aHeight, int aBytesPerLine) override
	{
		auto input = makeConstHostImageView(
			reinterpret_cast<const element_rgb8_t *>(aBuffer),
			Int2(aWidth, aHeight),
			Int2(sizeof(element_rgb8_t), aBytesPerLine));
		setStartImageView(input);
	}

	void
	getCurrentImage(unsigned char *aBuffer, int aWidth, int aHeight, int aBytesPerLine) override
	{
		std::cout << size_t(aBuffer) << "; " << aWidth << "; " << aBytesPerLine << "\n";
		auto result = makeHostImageView(
			reinterpret_cast<element_rgb8_t *>(aBuffer),
			Int2(aWidth,aHeight),
			Int2(sizeof(element_rgb8_t), aBytesPerLine));
		fillFromCurrentImage(result);
	}

	virtual void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView) = 0;

	virtual void
	fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView) = 0;
};

class ConwaysAutomatonWrapper: public AutomatonWrapper
{
public:
	ConwaysAutomatonWrapper()
	{}

	void
	runIterations(int aIterationCount) override;

	virtual void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView) override;

	virtual void
	fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView) override;

	CellularAutomaton<Grid<uint8_t, 2>, MooreNeighborhood<2>, ConwayRule> mAutomaton;
};


void
ConwaysAutomatonWrapper::runIterations(int aIterationCount)
{
	mAutomaton.iterate(aIterationCount);
}

void ConwaysAutomatonWrapper::setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView)
{
	host_image<uint8_t, 2> hostInput(aView.dimensions());
	copy(unaryOperator(aView, ColorToCell()), view(hostInput));
	mAutomaton.initialize(const_view(hostInput));
}

void ConwaysAutomatonWrapper::fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView)
{
	auto state = mAutomaton.getCurrentState();
	host_image<uint8_t, 2> hostState(state.dimensions());
	copy(state, view(hostState));
	copy(unaryOperator(const_view(hostState), CellToColor()), aView);
}

std::unique_ptr<AAutomatonWrapper>
getConwaysAutomatonWrapper() {
	return std::unique_ptr<AAutomatonWrapper>(new ConwaysAutomatonWrapper());
}

//******************************************************************************************

template<typename TAutomaton>
class CCLAutomatonWrapper: public AutomatonWrapper
{
public:
	void
	runIterations(int aIterationCount) override;

	virtual void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView) override;

	virtual void
	fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView) override;

	TAutomaton mAutomaton;
};


template<typename TAutomaton>
void
CCLAutomatonWrapper<TAutomaton>::runIterations(int aIterationCount)
{
	mAutomaton.iterate(aIterationCount);
}

template<typename TAutomaton>
void CCLAutomatonWrapper<TAutomaton>::setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView)
{
	host_image<int, 2> hostInput(aView.dimensions());
	copy(unaryOperatorOnPosition(aView, ColorToLabel(aView.dimensions())), view(hostInput));
	mAutomaton.initialize(const_view(hostInput));
}

template<typename TAutomaton>
void CCLAutomatonWrapper<TAutomaton>::fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView)
{
	auto state = mAutomaton.getCurrentState();
	host_image<int, 2> hostState(state.dimensions());
	copy(state, view(hostState));
	/*auto cv = const_view(hostState);
	for (int j = 0; j < cv.dimensions()[1]; ++j){
		for (int i = 0; i < cv.dimensions()[0]; ++i){
			std::cout << "\t" << cv[Int2(i, j)];
		}
		std::cout << "\n";
	}*/
	copy(unaryOperator(const_view(hostState), assign_color_ftor()), aView);
}

std::unique_ptr<AAutomatonWrapper>
getCCLAutomatonWrapper() {
	return std::unique_ptr<AAutomatonWrapper>(new CCLAutomatonWrapper<CellularAutomaton<Grid<int, 2>, VonNeumannNeighborhood<2>, ConnectedComponentLabelingRule>>());
}

typedef CellularAutomatonWithGlobalState<Grid<int, 2>, VonNeumannNeighborhood<2>, ConnectedComponentLabelingRule2, EquivalenceGlobalState> CCLMergeAutomaton;

class CCLAutomatonMergeWrapper: public CCLAutomatonWrapper<CCLMergeAutomaton>
{
public:
	void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView)
	{
		host_image<int, 2> hostInput(aView.dimensions());
		copy(unaryOperatorOnPosition(aView, ColorToLabel(aView.dimensions())), view(hostInput));

		EquivalenceGlobalState globalState;
		mBuffer.resize(elementCount(aView) + 1);
		globalState.manager = EquivalenceManager<int>(thrust::raw_pointer_cast(mBuffer.data()), mBuffer.size());
		mAutomaton.initialize(const_view(hostInput), globalState);
	}

	thrust::device_vector<int> mBuffer;
};

std::unique_ptr<AAutomatonWrapper>
getCCLAutomatonWrapper2() {
	return std::unique_ptr<AAutomatonWrapper>(new CCLAutomatonMergeWrapper());
}

//******************************************************************************************

class WShedAutomatonWrapper: public AutomatonWrapper
{
public:
	typedef Tuple<int, int, int> Value;
	typedef Tuple<int, int> Value2;
	//typedef simple_vector<int, 2> Value2;

	void
	runIterations(int aIterationCount) override;

	virtual void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView) override;

	virtual void
	fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView) override;

	CellularAutomatonWithGlobalState<Grid<Value2, 2>, VonNeumannNeighborhood<2>, LocalMinimaConnectedComponentRule, LocalMinimaEquivalenceGlobalState<int>> mLocalMinimumAutomaton;
	CellularAutomatonWithGlobalState<Grid<Value, 2>, MooreNeighborhood<2>, WatershedRule, ConvergenceFlag> mAutomaton;
	thrust::device_vector<int> mBuffer;
	device_flag convergenceFlag;
	ConvergenceFlag mConvergenceGlobalState;
};

void
WShedAutomatonWrapper::runIterations(int aIterationCount)
{
	//mLocalMinimumAutomaton.iterate(aIterationCount);
	mAutomaton.iterate(aIterationCount);
	std::cout << "CONVERGENCE FLAG " << mConvergenceGlobalState.is_finished() << std::endl;
	mConvergenceGlobalState.mDeviceFlag.reset_host();
}

struct InitWatershed
{
	CUGIP_DECL_HYBRID
	Tuple<int, int, int>
	operator()(int aGradient, int aLocalMinimum) const
	{
		return Tuple<int, int, int>(aGradient, aLocalMinimum, aLocalMinimum > 0 ? 0 : 999999);
	}
};

struct ZipGradientAndLabel
{
	CUGIP_DECL_HYBRID
	Tuple<int, int>
	operator()(int aGradient, int aLocalMinimum) const
	{
		return Tuple<int, int>(aGradient, aLocalMinimum);
	}
};

template<typename TView>
void computeGradientInplace(TView aView)
{
		device_image<int, 2> tmpImage(aView.dimensions());
		auto gradient = lowerLimit(0, 0, unaryOperatorOnLocator(aView, gradient_magnitude<int,int>()));
		copy(gradient, view(tmpImage));
		copy(const_view(tmpImage), aView);

}

void WShedAutomatonWrapper::setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView)
{
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	host_image<int, 2> hostGrayscale(aView.dimensions());
	copy(unaryOperator(aView, ChannelSumFunctor()/*grayscale_ftor()*/), view(hostGrayscale));
	device_image<int, 2> deviceGradient(aView.dimensions());
	copy(const_view(hostGrayscale), view(deviceGradient));

	if (this->mPreprocessingEnabled) {
		computeGradientInplace(view(deviceGradient));
	}
	auto localMinima = unaryOperatorOnLocator(const_view(deviceGradient), LocalMinimumLabel());

	LocalMinimaEquivalenceGlobalState<int> globalState;
	mBuffer.resize(elementCount(aView) + 1);
	globalState.manager = EquivalenceManager<int>(thrust::raw_pointer_cast(&mBuffer[0]), mBuffer.size());
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	mLocalMinimumAutomaton.initialize(nAryOperator(ZipGradientAndLabel(), const_view(deviceGradient), localMinima), globalState);
	mLocalMinimumAutomaton.iterate(500);

	auto wshed = nAryOperator(InitWatershed(), const_view(deviceGradient), getDimension(mLocalMinimumAutomaton.getCurrentState(), IntValue<1>()));

	//ConvergenceFlag convergenceGlobalState{ convergenceFlag.view() };
	mConvergenceGlobalState.mDeviceFlag = convergenceFlag.view();
	mAutomaton.initialize(wshed, mConvergenceGlobalState);
}

void WShedAutomatonWrapper::fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView)
{
	//auto state = mLocalMinimumAutomaton.getCurrentState();
	auto state = mAutomaton.getCurrentState();
	device_image<int, 2> tmpState(state.dimensions());
	host_image<int, 2> hostState(state.dimensions());
	copy(getDimension(state, IntValue<1>()), view(tmpState));
	copy(const_view(tmpState), view(hostState));
	copy(unaryOperator(const_view(hostState), assign_color_ftor()), aView);
}

std::unique_ptr<AAutomatonWrapper>
getWShedAutomatonWrapper() {
	return std::unique_ptr<AAutomatonWrapper>(new WShedAutomatonWrapper());
}


//******************************************************************************************

class WShedAutomaton2Wrapper: public AutomatonWrapper
{
public:
	typedef Tuple<int, int, int> Value;

	void
	runIterations(int aIterationCount) override;

	virtual void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView) override;

	virtual void
	fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView) override;

	CellularAutomaton<Grid<Value, 2>, MooreNeighborhood<2>, Watershed2Rule> mAutomaton;
	thrust::device_vector<int> mBuffer;
	device_flag convergenceFlag;
	Watershed2EquivalenceGlobalState globalState;
};


void
WShedAutomaton2Wrapper::runIterations(int aIterationCount)
{
	mAutomaton.iterate(aIterationCount);
}

void WShedAutomaton2Wrapper::setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView)
{
	mBuffer.resize(elementCount(aView) + 1);
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	host_image<int, 2> hostGrayscale(aView.dimensions());
	copy(unaryOperator(aView, ChannelSumFunctor()/*grayscale_ftor()*/), view(hostGrayscale));
	device_image<int, 2> deviceGradient(aView.dimensions());
	copy(const_view(hostGrayscale), view(deviceGradient));

	if (this->mPreprocessingEnabled) {
		computeGradientInplace(view(deviceGradient));
	}
	auto smallerNeighbor = unaryOperatorOnLocator(const_view(deviceGradient), HasSmallerNeighbor());

	//Watershed2EquivalenceGlobalState globalState;
	globalState.manager = EquivalenceManager<int>(thrust::raw_pointer_cast(mBuffer.data()), mBuffer.size());
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<2>(aView.dimensions()), smallerNeighbor);
	mAutomaton.initialize(wshed);
}

void WShedAutomaton2Wrapper::fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView)
{
	auto state = mAutomaton.getCurrentState();
	device_image<int, 2> tmpState(state.dimensions());
	copy(getDimension(state, IntValue<1>()), view(tmpState));

	host_image<int, 2> hostState(state.dimensions());
	copy(const_view(tmpState), view(hostState));
	copy(unaryOperator(const_view(hostState), assign_color_ftor()), aView);
}

std::unique_ptr<AAutomatonWrapper>
getWShedAutomaton2Wrapper() {
	return std::unique_ptr<AAutomatonWrapper>(new WShedAutomaton2Wrapper());
}


//******************************************************************************************

class WShedAutomaton2GlobalStateWrapper: public AutomatonWrapper
{
public:
	typedef Tuple<int, int, int> Value;

	void
	runIterations(int aIterationCount) override;

	virtual void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView) override;

	virtual void
	fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView) override;

	CellularAutomatonWithGlobalState<Grid<Value, 2>, MooreNeighborhood<2>, Watershed2GlobalStateRule, Watershed2EquivalenceGlobalState> mAutomaton;
	thrust::device_vector<int> mBuffer;
	device_flag convergenceFlag;
	Watershed2EquivalenceGlobalState globalState;
};


void
WShedAutomaton2GlobalStateWrapper::runIterations(int aIterationCount)
{
	mAutomaton.iterate(aIterationCount);
}

void WShedAutomaton2GlobalStateWrapper::setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView)
{
	mBuffer.resize(elementCount(aView) + 1);
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	host_image<int, 2> hostGrayscale(aView.dimensions());
	copy(unaryOperator(aView, ChannelSumFunctor()/*grayscale_ftor()*/), view(hostGrayscale));
	device_image<int, 2> deviceGradient(aView.dimensions());
	copy(const_view(hostGrayscale), view(deviceGradient));

	if (this->mPreprocessingEnabled) {
		computeGradientInplace(view(deviceGradient));
	}
	auto smallerNeighbor = unaryOperatorOnLocator(const_view(deviceGradient), HasSmallerNeighbor());

	//Watershed2EquivalenceGlobalState globalState;
	globalState.manager = EquivalenceManager<int>(thrust::raw_pointer_cast(mBuffer.data()), mBuffer.size());
	globalState.mDeviceFlag = convergenceFlag.view();
	globalState.manager.initialize();

	auto wshed = zipViews(const_view(deviceGradient), UniqueIdDeviceImageView<2>(aView.dimensions()), smallerNeighbor);
	mAutomaton.initialize(wshed, globalState);
}

void WShedAutomaton2GlobalStateWrapper::fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView)
{
	auto state = mAutomaton.getCurrentState();
	device_image<int, 2> tmpState(state.dimensions());
	copy(getDimension(state, IntValue<1>()), view(tmpState));

	host_image<int, 2> hostState(state.dimensions());
	copy(const_view(tmpState), view(hostState));
	copy(unaryOperator(const_view(hostState), assign_color_ftor()), aView);
}


//******************************************************************************************

class ReactionDiffusionWrapper: public AutomatonWrapper
{
public:
	typedef vect2f_t Value;

	void
	runIterations(int aIterationCount) override;

	virtual void
	setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView) override;

	virtual void
	fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView) override;

	CellularAutomaton<Grid<Value, 2>, VonNeumannNeighborhood<2>, ReactionDiffusionRule> mAutomaton;
};


void
ReactionDiffusionWrapper::runIterations(int aIterationCount)
{
	mAutomaton.iterate(aIterationCount);
}

struct ColorToReaction
{
	CUGIP_DECL_HYBRID
	vect2f_t
	operator()(element_rgb8_t value) const
	{
		return vect2f_t(value.data[0] / 255.0f, value.data[2] / 255.0f);
	}
};


void ReactionDiffusionWrapper::setStartImageView(const_host_image_view<const element_rgb8_t, 2> aView)
{
	host_image<Value, 2> hostReaction(aView.dimensions());
	copy(unaryOperator(aView, ColorToReaction()), view(hostReaction));
	/*device_image<Value, 2> (aView.dimensions());
	copy(const_view(hostGrayscale), view(deviceGradient));*/

	//mAutomaton.initialize(unaryOperator(aView, ColorToReaction()));
	mAutomaton.initialize(view(hostReaction));
}

struct ReactionToColor
{
	CUGIP_DECL_HYBRID
	element_rgb8_t
	operator()(vect2f_t value) const
	{
		value -= minVal;
		value = div(value, (maxVal - minVal));
		return element_rgb8_t{ uint8_t(255 * value[0]), 0, uint8_t(255 * value[1]) };
	}
	vect2f_t minVal;
	vect2f_t maxVal;
};

void ReactionDiffusionWrapper::fillFromCurrentImage(host_image_view<element_rgb8_t, 2> aView)
{
	auto state = mAutomaton.getCurrentState();

	auto maxVal = max(state);
	auto minVal = min(state);
	auto colorConverted = unaryOperator(state, ReactionToColor{minVal, maxVal});
	device_image<element_rgb8_t, 2> deviceImage(state.dimensions());

	copy(colorConverted, view(deviceImage));
	copy(const_view(deviceImage), aView);
}

std::unique_ptr<AAutomatonWrapper>
getReactionDiffusionWrapper() {
	return std::unique_ptr<AAutomatonWrapper>(new ReactionDiffusionWrapper());
}

template<typename TAutomatonWrapper>
std::unique_ptr<AAutomatonWrapper>
createAutomaton() {
	return std::unique_ptr<AAutomatonWrapper>(new TAutomatonWrapper());
}

const AutomatonFactoryMap & automatonFactoryMap() {
	static AutomatonFactoryMap factoryMap = {
		{"Conway's Game of Life", &createAutomaton<ConwaysAutomatonWrapper>},
		{"CCL", &createAutomaton<CCLAutomatonWrapper<CellularAutomaton<Grid<int, 2>, VonNeumannNeighborhood<2>, ConnectedComponentLabelingRule>>>},
		{"CCL with global state", &createAutomaton<CCLAutomatonMergeWrapper>},
		{"Watersheds", &createAutomaton<WShedAutomatonWrapper>},
		{"Watersheds 2", &createAutomaton<WShedAutomaton2Wrapper>},
		{"Watersheds 2 with global state", &createAutomaton<WShedAutomaton2GlobalStateWrapper>},
		{"Reaction-Diffusion", &createAutomaton<ReactionDiffusionWrapper>},
	};

	return factoryMap;
}
