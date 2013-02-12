
#include <boost/gil/gil_all.hpp>
#include <cugip/image.hpp>
#include <cugip/copy.hpp>
#include <cugip/for_each.hpp>
#include <cugip/functors.hpp>

void
process(boost::gil::rgb8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut)
{

	cugip::device_image<cugip::element_rgb8_t> inImage(aIn.width(), aIn.height());


	cugip::copy(aIn, cugip::view(inImage));

	cugip::for_each(cugip::view(inImage), cugip::negate<cugip::element_rgb8_t>());

	cugip::copy(cugip::view(inImage), aOut);
}
