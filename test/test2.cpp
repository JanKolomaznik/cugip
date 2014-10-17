
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>

using namespace boost::gil;

void
negative(rgb8_image_t::const_view_t aIn, rgb8_image_t::view_t aOut);
void
grayscale(boost::gil::rgb8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);

void
thresholding(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);

void
colored_ccl(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut);

void
colored_watersheds(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut);

void
mandelbrot(boost::gil::rgb8_image_t::view_t aOut);
void
gradient(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut);
void
gradient_mag(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);

void
diffusion(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);

void
laplacian(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);


void
generate_spiral(boost::gil::gray8_image_t::view_t aOut);

void
generate_basins(boost::gil::gray8_image_t::view_t aOut);


int main() {
	//gray8_image_t img;
	//jpeg_read_image("test.jpg",img);

	{
		gray8_image_t img(400,400);
		generate_spiral(view(img));


		rgb8_image_t colored_ccl_out(img.dimensions());
		colored_ccl(const_view(img), view(colored_ccl_out));
		//jpeg_write_view("colored_ccl_out.jpg",const_view(colored_ccl_out));
		png_write_view("colored_ccl_out.png",const_view(colored_ccl_out));
	}

/*	{
		gray8_image_t img(2000, 2000);
		generate_basins(view(img));

		rgb8_image_t colored_wshed_out(img.dimensions());
		colored_watersheds(const_view(img), view(colored_wshed_out));
		//jpeg_write_view("colored_ccl_out.jpg",const_view(colored_ccl_out));
		png_write_view("colored_wshed_out.png",const_view(colored_wshed_out));
	}*/
	return 0;
}
