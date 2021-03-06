
#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__


#include <boost/gil/gil_all.hpp>
#include <cugip/utils.hpp>
#include <cugip/image.hpp>
#include <cugip/transform.hpp>
#include <cugip/copy.hpp>
#include <cugip/for_each.hpp>
#include <cugip/filter.hpp>
#include <cugip/functors.hpp>
#include <cugip/exception.hpp>
#include <cugip/union_find.hpp>
#include <cugip/basic_filters/convolution.hpp>
#include <cugip/basic_filters/gradient.hpp>
#include <cugip/basic_filters/connected_component_labeling.hpp>
#include <cugip/basic_filters/local_minima_detection.hpp>
#include <cugip/basic_filters/watershed_transformation.hpp>
#include <cugip/algebra/arithmetics.hpp>
#include <cugip/advanced_operations/coherence_enhancing_diffusion.hpp>
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>


#include <boost/timer/timer.hpp>


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
thresholding(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_gray8_t> outImage(aIn.width(), aIn.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::transform(cugip::view(inImage), cugip::view(outImage), cugip::thresholding_ftor<cugip::element_gray8_t, cugip::element_gray8_t>(80, 255, 0));

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
colored_ccl(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
	D_PRINT("Colored CCL");
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<int> ids(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	cugip::device_union_find<int> lut(aIn.width() * aIn.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::assign_masked_ids(cugip::const_view(inImage), cugip::view(ids));
	cugip::connected_component_labeling(cugip::view(ids), cugip::view(lut));
	cugip::transform(cugip::const_view(ids), cugip::view(outImage), cugip::assign_color_ftor());

	cugip::copy(cugip::view(outImage), aOut);
	D_PRINT(cugip::cudaMemoryInfoText());
	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
colored_minima(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
	D_PRINT("Colored minima");
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<int> ids(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	cugip::device_union_find<int> lut(aIn.width() * aIn.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::local_minima_detection(cugip::const_view(inImage), cugip::view(ids), cugip::view(lut));
	cugip::transform(cugip::const_view(ids), cugip::view(outImage), cugip::assign_color_ftor());

	cugip::copy(cugip::view(outImage), aOut);
	D_PRINT(cugip::cudaMemoryInfoText());
	CUGIP_CHECK_ERROR_STATE("CHECK");
}



void
colored_watersheds1(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
	D_PRINT("Colored WSheds");
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<int> ids(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	cugip::device_union_find<int> lut(aIn.width() * aIn.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	{
		boost::timer::auto_cpu_timer t;
		cugip::watershed_transformation1(
			cugip::const_view(inImage), 
			cugip::view(ids),
			cugip::view(lut));
		cudaThreadSynchronize();
	}
	cugip::transform(cugip::const_view(ids), cugip::view(outImage), cugip::assign_color_ftor());

	cugip::copy(cugip::view(outImage), aOut);
	D_PRINT(cugip::cudaMemoryInfoText());
	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
colored_watersheds2(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
	D_PRINT("Colored WSheds");
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<int> ids(aIn.width(), aIn.height());
	cugip::device_image<int> ids2(aIn.width(), aIn.height());
	cugip::device_image<int> distance1(aIn.width(), aIn.height());
	cugip::device_image<int> distance2(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	cugip::device_union_find<int> lut(aIn.width() * aIn.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	{
		boost::timer::auto_cpu_timer t;
		cugip::local_minima_detection(cugip::const_view(inImage), cugip::view(ids), cugip::view(lut));
		{
			boost::timer::auto_cpu_timer t2;
			cugip::watershed_transformation2(
				cugip::const_view(inImage), 
				cugip::view(ids),
				cugip::view(ids2),
				cugip::view(distance1),
				cugip::view(distance2));
			cudaThreadSynchronize();
		}
	}
	cugip::transform(cugip::const_view(ids), cugip::view(outImage), cugip::assign_color_ftor());

	cugip::copy(cugip::view(outImage), aOut);
	D_PRINT(cugip::cudaMemoryInfoText());
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
	boost::timer::auto_cpu_timer t;
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


	CUGIP_CHECK_ERROR_STATE("CHECK");
}


void
generate_spiral(boost::gil::gray8_image_t::view_t aOut)
{
	boost::timer::auto_cpu_timer t;
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	{
		boost::timer::auto_cpu_timer t2;
		cugip::for_each_position(cugip::view(outImage), cugip::generate_spiral_ftor(outImage.dimensions()));
		cudaThreadSynchronize();
	}
	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}



void
generate_basins(boost::gil::gray8_image_t::view_t aOut)
{
	boost::timer::auto_cpu_timer t;
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	{
		boost::timer::auto_cpu_timer t2;
		cugip::for_each_position(cugip::view(outImage), cugip::generate_basins_ftor());
		cudaThreadSynchronize();
	}
	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}




void
gradient(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::filter(cugip::const_view(inImage), cugip::view(outImage), cugip::gradient_symmetric_difference<cugip::element_gray8_t, cugip::element_rgb8_t>());

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
gradient_mag(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::filter(
		cugip::const_view(inImage), 
		cugip::view(outImage), 
		cugip::gradient_magnitude_symmetric_difference<cugip::element_gray8_t, cugip::element_gray8_t>());

	cugip::multiply(cugip::view(outImage), 15);

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}

void
diffusion(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<float> tmpImage(aIn.width(), aIn.height());
	cugip::device_image<float> diffStep(aIn.width(), aIn.height());
	cugip::device_image<cugip::simple_vector<float, 2> > gradient(aIn.width(), aIn.height());
	cugip::device_image<cugip::simple_vector<float, 3> > structuralTensor(aOut.width(), aOut.height());
	cugip::device_image<cugip::simple_vector<float, 3> > diffusionTensor(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::transform(cugip::const_view(inImage), cugip::view(tmpImage), cugip::convert_float_and_byte());

	for (int i = 0; i < 80; ++i) {
		coherence_enhancing_diffusion_step(
			cugip::const_view(tmpImage), 
			cugip::view(diffStep), 
			cugip::view(gradient), 
			cugip::view(structuralTensor), 
			cugip::view(diffusionTensor)
			);

		cugip::add(cugip::view(tmpImage), 0.1f, cugip::const_view(diffStep));
	
		cugip::transform(cugip::const_view(tmpImage), cugip::view(inImage), cugip::convert_float_and_byte());
		
		if (i%5 == 4) 
		{
			cugip::copy(cugip::view(inImage), aOut);
			CUGIP_CHECK_ERROR_STATE("CHECK");
			std::string path = boost::str(boost::format("diffusion_%|03|.jpg") % i);
			boost::gil::jpeg_write_view(path.c_str(), aOut);
		}
	}
}

void
laplacian(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut)
{
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_gray8_t> outImage(aOut.width(), aOut.height());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(aIn, cugip::view(inImage));

	cugip::convolution(cugip::const_view(inImage), cugip::view(outImage), cugip::laplacian_kernel(), 128);

	cugip::copy(cugip::view(outImage), aOut);

	CUGIP_CHECK_ERROR_STATE("CHECK");
}


void
test(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{
	/*D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<cugip::element_gray8_t> inImage(aIn.width(), aIn.height());
	cugip::device_image<int> ids(aIn.width(), aIn.height());
	cugip::device_image<cugip::element_rgb8_t> outImage(aOut.width(), aOut.height());

	connected_component_labeling(cugip::const_view(inImage), cugip::view(ids));
	cugip::transform(cugip::const_view(ids), cugip::view(outImage), cugip::assign_color_ftor());

	D_PRINT(cugip::cudaMemoryInfoText());
*/
}

