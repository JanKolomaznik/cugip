
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>

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
mandelbrot(boost::gil::rgb8_image_t::view_t aOut);
void
gradient(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::rgb8_image_t::view_t aOut);
void
gradient_mag(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);

void
diffusion(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);

void
laplacian(boost::gil::gray8_image_t::const_view_t aIn, boost::gil::gray8_image_t::view_t aOut);

int main() {
    rgb8_image_t img;
    jpeg_read_image("test.jpg",img);

    gray8_image_t gray_out(img.dimensions());
    grayscale(const_view(img), view(gray_out));
    jpeg_write_view("gray_out.jpg",const_view(gray_out));

    gray8_image_t thresholding_out(img.dimensions());
    thresholding(const_view(gray_out), view(thresholding_out));

    rgb8_image_t colored_ccl_out(img.dimensions());
    colored_ccl(const_view(thresholding_out), view(colored_ccl_out));
    jpeg_write_view("colored_ccl_out.jpg",const_view(colored_ccl_out));

    return 0;
}
