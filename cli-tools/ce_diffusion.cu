#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

//#include "itkImage.h"

#include <cugip/image.hpp>
#include <cugip/advanced_operations/coherence_enhancing_diffusion.hpp>
#include <cugip/memory_view.hpp>
#include <cugip/memory.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/basic_filters/convolution.hpp>
#include <cugip/timers.hpp>

//typedef itk::Image<float, 3> ImageType;
using namespace cugip;

template<int tDimension>
struct sobel_gradient_magnitude
{
	sobel_gradient_magnitude()
		: kernel(sobel_gradient_kernel<tDimension>())
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID float
	operator()(TLocator aLocator) const
	{
		simple_vector<float, tDimension> result;
		apply_convolution_kernel(aLocator, kernel, result);
		return magnitude(result);
	}
	convolution_kernel<simple_vector<float, tDimension>, typename FillStaticSize<tDimension, 3>::Type> kernel;
};

void
coherenceEnhancingDiffusion(float *aInput, float *aOutput, size_t aWidth, size_t aHeight, size_t aDepth, int aIterationCount)
{
	auto inView = makeConstHostImageView(aInput, vect3i_t(aWidth, aHeight, aDepth));
	auto outView = makeHostImageView(aOutput, vect3i_t(aWidth, aHeight, aDepth));
	D_PRINT(aDepth);
	D_PRINT(inView.dimensions());

	//cugip::print_error_enums();
	D_PRINT(cugip::cudaMemoryInfoText());
	std::array<cugip::device_image<float, 3>, 2> images = {
		cugip::device_image<float, 3>(inView.dimensions()),
		cugip::device_image<float, 3>(inView.dimensions()) };
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(inView, cugip::view(images[0]));

	coherence_enhancing_diffusion<3> ceDiffusion(vect3i_t(aWidth, aHeight, aDepth), 0.05f, 0.001f, 15.0f);

	int i = 0;
	AggregatingTimerSet<1, int> timer;
	for (i = 0; i < aIterationCount; ++i) {
		auto interval = timer.start(0, i);
		ceDiffusion.iteration(const_view(images[i % 2]), view(images[(i+1) % 2]));
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}
	std::cout << timer.createReport({"\nCED iterations"});

	cugip::copy(cugip::const_view(images[i % 2]), outView);

	CUGIP_CHECK_ERROR_STATE("CED");

}
