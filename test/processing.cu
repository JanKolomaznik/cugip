
#include <boost/gil/gil_all.hpp>
#include <cugip/image.hpp>
#include <cugip/transform.hpp>
#include <cugip/copy.hpp>
#include <cugip/for_each.hpp>
#include <cugip/filter.hpp>
#include <cugip/functors.hpp>
#include <cugip/exception.hpp>

void
negative(boost::gil::rgb8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_rgb8_t> inImage(aIn.width(), aIn.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::for_each(cugip::view(inImage), cugip::negate<cugip::element_rgb8_t>());

	cugip::copy(cugip::view(inImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
grayscale(boost::gil::rgb8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_rgb8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::transform(cugip::const_view(inImage), cugip::view(outImage), cugip::grayscale_ftor());

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
mandelbrot(boost::gil::rgb8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::for_each_position(cugip::view(outImage), 
			cugip::mandelbrot_ftor(outImage.dimensions(), 
			cugip::intervalf_t(-0.95f, -0.85f), 
			cugip::intervalf_t(-0.3f, -0.25f)));

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
gradient(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

//	cugip::filter(cugip::const_view(inImage), cugip::view(outImage), cugip::gradient_sobel<cugip::element_gray8_t, cugip::element_gray8_t>());
	cugip::filter(cugip::const_view(inImage), cugip::view(outImage), cugip::gradient_difference<cugip::element_gray8_t, cugip::element_gray8_t>());

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}
