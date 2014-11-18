
#include <cugip/image.hpp>
#include <cugip/advanced_operations/nonlocal_means.hpp>


void
denoise(...aInput, ...aOutput)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<float> inImage(aInput.dimensions());
	cugip::device_image<float> outImage(aInput.dimensions());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aInput, cugip::view(inImage));

	cugip::nonlocal_means(cugip::const_view(inImage), cugip::view(outImage));

	cugip::copy(cugip::view(outImage), aOutput);

	CUGIP_CHECK_ERROR_STATE("denoise");

}


