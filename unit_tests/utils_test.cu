#define BOOST_TEST_MODULE UtilsTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/tuple.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/access_utils.hpp>
#include <cugip/math/symmetric_tensor.hpp>
#include <cugip/math/eigen.hpp>
#include <cugip/math/matrix.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>

using namespace cugip;

BOOST_AUTO_TEST_CASE(BlockedOrderAccessIndex)
{
	using namespace cugip;
	BOOST_CHECK_EQUAL(0, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(0, 0, 0)));
	BOOST_CHECK_EQUAL(Int3(5, FillFlag()), Int3(5, 5, 5));
}
