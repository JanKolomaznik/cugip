
#include <utility>

#include <cugip/image.hpp>
#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/math.hpp>
#include <cugip/tuple.hpp>
#include <cugip/for_each.hpp>
#include <cugip/transform.hpp>

#include <cugip/timers.hpp>
using namespace cugip;

struct PlusFunctor {
	template<typename T1, typename T2>
	CUGIP_DECL_HYBRID
	auto operator()(const Tuple<T1, T2> &aTup) const -> decltype(std::declval<T1>() + std::declval<T2>())
	{
		return get<0>(aTup) + get<1>(aTup);
	}
};

struct Polynom {
	device_image<float, 3> in1;
	device_image<float, 3> in2;
	device_image<float, 3> result;

	Polynom():
		in1(512, 512, 512),
		in2(512, 512, 512),
		result(512, 512, 512)
	{}

	void init()
	{
		auto checkers1 = checkerBoard(20.0f, -10.0f, Int3(7,6,5), Int3(512, 512, 512));
		auto checkers2 = checkerBoard(20.0f, -10.0f, Int3(3,8,11), Int3(512, 512, 512));

		copy(checkers1, view(in1));
		copy(checkers2, view(in2));
	}

	void lazyEvaluation()
	{
		auto computation = add(const_view(in1), squareRoot(const_view(in2)));
		copy(computation, view(result));
	}

	void separateKernels()
	{
		transform(const_view(in1), view(in1), SquareRootFunctor());
		transform(zipViews(const_view(in1), const_view(in2)), view(result), PlusFunctor());
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}

	void singleFunctor()
	{
		init();
		auto computation = add(const_view(in1), squareRoot(const_view(in2)));

		copy(computation, view(result));
	}

	/*void lazyEvaluation()
	{
		init();
		AggregatingTimerSet<1> timer;
		for (int i = 0; i < 100; ++i) {
			auto interval = timer.start<0>(0);
			auto computation = add(const_view(in1), squareRoot(const_view(in2)));

			copy(computation, view(result));
		}
			CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		std::cout << timer.createReport({"lazyEvaluation"}) << '\n';
	}*/
};

template<typename TImplementation>
struct Runner
{
	void lazyEvaluation() {
		implementation.init();
		AggregatingTimerSet<1> timer;
		for (int i = 0; i < iterationCount; ++i) {
			auto interval = timer.start<0>(0);
			implementation.lazyEvaluation();
		}
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		std::cout << timer.createReport({"lazyEvaluation"}) << '\n';
	}

	void separateKernels() {
		implementation.init();
		AggregatingTimerSet<1> timer;
		for (int i = 0; i < iterationCount; ++i) {
			auto interval = timer.start<0>(0);
			implementation.separateKernels();
		}
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		std::cout << timer.createReport({"separateKernels"}) << '\n';
	}

	TImplementation implementation;
	int iterationCount = 100;
};


int main(int argc, char **argv)
{
	try {
		Runner<Polynom> benchmark;
		std::cout << cudaMemoryInfoText() << '\n';
		benchmark.lazyEvaluation();

		std::cout << cudaMemoryInfoText() << '\n';
		benchmark.separateKernels();

		std::cout << cudaMemoryInfoText() << '\n';

	} catch (std::exception &e) {
		std::cout << boost::diagnostic_information(e) << '\n';
		throw;
	}
	return 0;
}
