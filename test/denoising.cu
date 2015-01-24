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

//typedef itk::Image<float, 3> ImageType;

void
denoise(float *aInput, float *aOutput, size_t aWidth, size_t aHeight, size_t aDepth, float aVariance)
//denoise(ImageType::Pointer aInput, ImageType::Pointer aOutput)
{
	//cugip::const_host_memory_3d<float> pom(aInput, aWidth, aHeight, aDepth, aWidth * sizeof(float));

	cugip::const_memory_view<float, 3> inView(cugip::const_host_memory_3d<float>(aInput, aWidth, aHeight, aDepth, aWidth * sizeof(float)));
	cugip::memory_view<float, 3> outView(cugip::host_memory_3d<float>(aOutput, aWidth, aHeight, aDepth, aWidth * sizeof(float)));
	D_PRINT(aDepth);
	D_PRINT(inView.dimensions());

	//cugip::print_error_enums();
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<float, 3> inImage(inView.dimensions());
	cugip::device_image<float, 3> outImage(inView.dimensions());
	D_PRINT(cugip::cudaMemoryInfoText());

	D_PRINT("nonlocal_means ...");

	cugip::copy_to(inView, cugip::view(inImage));

	cugip::nonlocal_means(cugip::const_view(inImage), cugip::view(outImage), cugip::nl_means_parameters<2, 3>(aVariance));

	cugip::copy_from(cugip::view(outImage), outView);
	D_PRINT("nonlocal_means done!");

	CUGIP_CHECK_ERROR_STATE("denoise");

}


