
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
    
    rgb8_image_t negative_out(img.dimensions());
    negative(const_view(img), view(negative_out));
    jpeg_write_view("negative_out.jpg",const_view(negative_out));

    gray8_image_t gray_out(img.dimensions());
    grayscale(const_view(img), view(gray_out));
    jpeg_write_view("gray_out.jpg",const_view(gray_out));


    gray8_image_t thresholding_out(img.dimensions());
    thresholding(const_view(gray_out), view(thresholding_out));
    jpeg_write_view("thresholding_out.jpg",const_view(thresholding_out));

    rgb8_image_t colored_ccl_out(img.dimensions());
    colored_ccl(const_view(thresholding_out), view(colored_ccl_out));
    jpeg_write_view("colored_ccl_out.jpg",const_view(colored_ccl_out));

    /*rgb8_image_t gradient_out(img.dimensions());
    gradient(const_view(gray_out), view(gradient_out));
    jpeg_write_view("gradient_out.jpg",const_view(gradient_out));

    gray8_image_t gradient_mag_out(img.dimensions());
    gradient_mag(const_view(gray_out), view(gradient_mag_out));
    jpeg_write_view("gradient_mag_out.jpg",const_view(gradient_mag_out));

    gray8_image_t laplacian_out(img.dimensions());
    laplacian(const_view(gray_out), view(laplacian_out));
    jpeg_write_view("laplacian_out.jpg",const_view(laplacian_out));

    gray8_image_t diffusion_out(img.dimensions());
    diffusion(const_view(gradient_mag_out), view(diffusion_out));
    jpeg_write_view("diffusion_out.jpg",const_view(diffusion_out));

    rgb8_image_t mandelbrot_out(1600,800);
    mandelbrot( view(mandelbrot_out));
    jpeg_write_view("mandelbrot_out.jpg",const_view(mandelbrot_out));*/

    return 0;
}
