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
/*#include <cugip/subview.hpp>
#include <cugip/view_arithmetics.hpp>*/
#include <cugip/transform.hpp>
/*#include <thrust/device_vector.h>
#include <thrust/reduce.h>*/

using namespace cugip;

static const float cEpsilon = 0.00001;

struct TestTransformFunctor
{
	template<typename TLocator>
	CUGIP_DECL_HYBRID int //typename TLocator::value_type
	operator()(TLocator aLocator) const
	{
		return aLocator[Int3(-1, FillFlag())] + aLocator[Int3(1, FillFlag())];
	}
};

BOOST_AUTO_TEST_CASE(TransformDevice)
{

	device_image<int, 3> deviceImage1(300, 300, 4);
	device_image<int, 3> deviceImage2(300, 300, 4);

	auto input = constantImage(25, deviceImage1.dimensions());

	copy(input, view(deviceImage1));

	transform(const_view(deviceImage1), view(deviceImage2), []__device__(const int &value) { return value + 1; });

	auto difference = sum_differences(const_view(deviceImage2), constantImage(26, deviceImage1.dimensions()), 0);
	BOOST_CHECK_EQUAL(difference, 0);
}

BOOST_AUTO_TEST_CASE(BoundedTransformWithPreload)
{

	device_image<int, 3> deviceImage1(30, 30, 4);
	device_image<int, 3> deviceImage2(30, 30, 4);

	auto input = UniqueIdDeviceImageView<3>(deviceImage1.dimensions());

	transform_locator(input, view(deviceImage1), TestTransformFunctor());
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	transform_locator(input, view(deviceImage2), TestTransformFunctor(), PreloadingTransformLocatorPolicy<decltype(input), 1>());
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	auto difference = sum_differences(view(deviceImage1), view(deviceImage2), 0);
	BOOST_CHECK_EQUAL(difference, 0);
}
