
#include <boost/gil/gil_all.hpp>
#include <cugip/image.hpp>

void
process(boost::gil::rgb8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{

	cugip::device_image<cugip::element_rgb8_t> inImage(aIn.width(), aIn.height());


	cugip::copy(aIn, cugip::view(inImage));

	cugip::for_each(cugip::view(inImage), negate());

	cugip::copy(cugip::view(inImage), aOut);
}
