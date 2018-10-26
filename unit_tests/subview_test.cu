#define BOOST_TEST_MODULE SubviewTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/image.hpp>
#include <cugip/image_dumping.hpp>
#include <cugip/host_image.hpp>
#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/subview.hpp>
#include <cugip/reduce.hpp>

using namespace cugip;

//static const float cEpsilon = 0.00001;


BOOST_AUTO_TEST_CASE(Subview)
{
	device_image<int, 2> deviceImage(128, 128);

	auto tiles = checkerBoard(0, 1, vect2i_t(5, 5), deviceImage.dimensions());
	copy(tiles, view(deviceImage));


	auto subview1 = subview(const_view(deviceImage), vect2i_t(7, 7), vect2i_t(40, 40));
	auto subview2 = subview(tiles, vect2i_t(7, 7), vect2i_t(40, 40));

	auto difference = sum_differences(subview1, subview2, int{0});
	BOOST_CHECK_EQUAL(difference, 0);
}

BOOST_AUTO_TEST_CASE(Slice)
{
	device_image<int, 3> deviceImage(128, 128, 128);

	auto tiles = checkerBoard(0, 1, vect3i_t(5, 5, 5), deviceImage.dimensions());
	copy(tiles, view(deviceImage));


	auto slice1 = slice<1>(const_view(deviceImage), 35);
	auto slice2 = slice<1>(tiles, 35);

	auto difference = sum_differences(slice1, slice2, int{0});
	BOOST_CHECK_EQUAL(difference, 0);
}

//TODO other tests
