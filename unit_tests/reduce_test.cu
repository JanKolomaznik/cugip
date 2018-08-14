#define BOOST_TEST_MODULE ReduceTest
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

using namespace cugip;

static const float cEpsilon = 0.00001;


BOOST_AUTO_TEST_CASE(ReduceSumConstantImage)
{
	device_image<int, 2> deviceImage(128, 128);
	device_image<int, 2> tmpImage(128, 128);

	auto input = constantImage(1, vect2i_t(128, 128));

	auto result = sum(input, 0);
	auto result2 = sum(input);
	
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());


	BOOST_CHECK_EQUAL(result, 128*128);
	BOOST_CHECK_EQUAL(result2, 128*128);
}

