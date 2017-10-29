
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>

using namespace boost::gil;

void
generate_spiral(boost::gil::gray8_image_t::view_t aOut);

void
generate_basins(boost::gil::gray8_image_t::view_t aOut);

int main() {
	gray8_image_t spiral_out(30, 30);
	generate_spiral(view(spiral_out));
	jpeg_write_view("spiral.jpg", const_view(spiral_out));

	gray8_image_t basins_out(1200, 1200);
	generate_basins(view(basins_out));
	jpeg_write_view("basins.jpg", const_view(basins_out));
	return 0;
}
