#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

//#include "itkImage.h"

#include <cugip/image.hpp>
#include <cugip/basic_filters/convolution.hpp>
#include <cugip/memory_view.hpp>
#include <cugip/memory.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>

//typedef itk::Image<float, 3> ImageType;
using namespace cugip;

void
gauss(float *aInput, float *aOutput, size_t aWidth, size_t aHeight, size_t aDepth, float aSigma)
{
	auto inView = makeConstHostImageView(aInput, vect3i_t(aWidth, aHeight, aDepth));
	auto outView = makeHostImageView(aOutput, vect3i_t(aWidth, aHeight, aDepth));
	D_PRINT(aDepth);
	D_PRINT(inView.dimensions());

	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<float, 3> inImage(inView.dimensions());
	cugip::device_image<float, 3> tmpImage(inView.dimensions());
	cugip::device_image<float, 3> outImage(inView.dimensions());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(inView, cugip::view(inImage));

	cugip::separable_convolution(cugip::const_view(inImage), cugip::view(outImage), cugip::view(tmpImage), gaussian_kernel<5>());

	cugip::copy(cugip::view(outImage), outView);

	CUGIP_CHECK_ERROR_STATE("gauss");
}
