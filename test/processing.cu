
#include <boost/gil/gil_all.hpp>
#include <cugip/image.hpp>
#include <cugip/transform.hpp>
#include <cugip/copy.hpp>
#include <cugip/for_each.hpp>
#include <cugip/filter.hpp>
#include <cugip/functors.hpp>
#include <cugip/exception.hpp>
#include <cugip/basic_filters/convolution.hpp>
#include <cugip/basic_filters/gradient.hpp>
#include <cugip/advanced_operations/coherence_enhancing_diffusion.hpp>


#include <boost/timer/timer.hpp>


void
negative(boost::gil::rgb8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
/*	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_rgb8_t> inImage(aIn.width(), aIn.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::for_each(cugip::view(inImage), cugip::negate<cugip::element_rgb8_t>());

	cugip::copy(cugip::view(inImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");*/

	//TODO - remove
	const cugip::element<int, 3> pom1 = {0};
	int i = cugip::get<1, const cugip::element<int, 3> >(pom1);// = 3;

}

void
grayscale(boost::gil::rgb8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
/*	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_rgb8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::transform(cugip::const_view(inImage), cugip::view(outImage), cugip::grayscale_ftor());

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");*/
}

void
mandelbrot(boost::gil::rgb8_image_t::view_t aOut)
{
/*	boost::timer::auto_cpu_timer t;
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	{
		boost::timer::auto_cpu_timer t2;
	cugip::for_each_position(cugip::view(outImage), 
			cugip::mandelbrot_ftor(outImage.dimensions(), 
			cugip::intervalf_t(-0.95f, -0.85f), 
			cugip::intervalf_t(-0.3f, -0.25f)));
		cudaThreadSynchronize();
	}
	cugip::copy(cugip::view(outImage), aOut);


	CUGIP_CHECK_ERROR_STATE("CHECK");*/
}

void
gradient(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
/*	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

//	cugip::filter(cugip::const_view(inImage), cugip::view(outImage), cugip::gradient_sobel<cugip::element_gray8_t, cugip::element_gray8_t>());
//	cugip::filter(cugip::const_view(inImage), cugip::view(outImage), cugip::gradient_difference<cugip::element_gray8_t, cugip::element_gray8_t>());
	cugip::filter(cugip::const_view(inImage), cugip::view(outImage), cugip::gradient_symmetric_difference<cugip::element_gray8_t, cugip::element_rgb8_t>());

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");*/
}

void
diffusion(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
/*	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::simple_vector<float, 2> > gradient(aIn.width(), aIn.height());
	cugip::device_image<cugip::simple_vector<float, 3> > structuralTensor(aOut.width(), aOut.height());
	cugip::device_image<cugip::simple_vector<float, 3> > diffusionTensor(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::compute_gradient(cugip::const_view(inImage), cugip::view(gradient));

	cugip::compute_structural_tensor(cugip::const_view(gradient), cugip::view(structuralTensor));

	cugip::blur_structural_tensor(cugip::const_view(structuralTensor), cugip::view(diffusionTensor));

	cugip::compute_diffusion_tensor(cugip::view(diffusionTensor));

	//cugip::copy(cugip::view(tensor), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");*/
}

void
laplacian(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
/*	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::convolution(cugip::const_view(inImage), cugip::view(outImage), cugip::laplacian_kernel(), 128);

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");*/
}
