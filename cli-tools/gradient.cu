#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

//#include "itkImage.h"

#include <cugip/image.hpp>
#include <cugip/advanced_operations/nonlocal_means.hpp>
#include <cugip/memory_view.hpp>
#include <cugip/memory.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/basic_filters/convolution.hpp>
#include <cugip/basic_filters/gradient.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/reduce.hpp>
#include <cugip/for_each.hpp>

//typedef itk::Image<float, 3> ImageType;
using namespace cugip;

void
gradientMagnitude(float *aInput, float *aOutput, size_t aWidth, size_t aHeight, size_t aDepth, bool aNormalize)
{
	//cugip::const_host_memory_3d<float> pom(aInput, aWidth, aHeight, aDepth, aWidth * sizeof(float));

	//cugip::const_memory_view<float, 3> inView(cugip::const_host_memory_3d<float>(aInput, aWidth, aHeight, aDepth, aWidth * sizeof(float)));
	//cugip::memory_view<float, 3> outView(cugip::host_memory_3d<float>(aOutput, aWidth, aHeight, aDepth, aWidth * sizeof(float)));
	auto inView = makeConstHostImageView(aInput, vect3i_t(aWidth, aHeight, aDepth));
	auto outView = makeHostImageView(aOutput, vect3i_t(aWidth, aHeight, aDepth));
	D_PRINT(aDepth);
	D_PRINT(inView.dimensions());

	//cugip::print_error_enums();
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<float, 3> inImage(inView.dimensions());
	cugip::device_image<float, 3> outImage(inView.dimensions());
	D_PRINT(cugip::cudaMemoryInfoText());

	//auto testView = checkerBoard(0.0f, 10.0f, vect3i_t(30, 30, 10), inView.dimensions());

	cugip::copy(inView, cugip::view(inImage));

	cugip::transform_locator(cugip::const_view(inImage), cugip::view(outImage), sobel_gradient_magnitude<3>());

	if (aNormalize) {
		auto maxVal = max(cugip::const_view(outImage));
		for_each(cugip::view(outImage), MultiplyByFactorFunctor<float>(100.0f / maxVal));
	}

	cugip::copy(cugip::view(outImage), outView);

	CUGIP_CHECK_ERROR_STATE("gradient");

}
