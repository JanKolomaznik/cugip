#define BOOST_TEST_MODULE ProceduralViewTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/reduce.hpp>
#include <cugip/math.hpp>
#include <cugip/image.hpp>
#include <cugip/host_image.hpp>
#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/for_each.hpp>
/*#include <cugip/subview.hpp>
#include <cugip/view_arithmetics.hpp>*/
#include <cugip/transform.hpp>
/*#include <thrust/device_vector.h>
#include <thrust/reduce.h>*/

using namespace cugip;

static const float cEpsilon = 0.00001;


BOOST_AUTO_TEST_CASE(ForEachDevice)
{

	device_image<int, 3> deviceImage1(320, 87, 45);

	auto input = constantImage(25, deviceImage1.dimensions());

	copy(input, view(deviceImage1));

	for_each(view(deviceImage1), []__device__(int &value) { ++value; });

	auto difference = sum_differences(view(deviceImage1), constantImage(26, deviceImage1.dimensions()), 0);
	BOOST_CHECK_EQUAL(difference, 0);

}


/*BOOST_AUTO_TEST_CASE(DeviceUtils)
{

	vect3i_t extents(30, 8, 4);
	vect3i_t block(32, 4, 4);

	int index = 1;
	
	auto corner = product(block, index_from_linear_access_index(div_up(extents, block), index));
	std::cout << "Corner " << corner << "\n";
	std::cout << "div_up(extents, block) " << div_up(extents, block) << "\n";
	std::cout << "index_from_linear_access_index(div_up(extents, block), index) " << index_from_linear_access_index(div_up(extents, block), index) << "\n";
	std::cout << "block " << block << "\n";
	BOOST_CHECK_EQUAL(corner, vect3i_t(0, 0, 4));

}*/

