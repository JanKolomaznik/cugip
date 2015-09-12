#define BOOST_TEST_MODULE ProceduralViewTest
#include <boost/test/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/reduce.hpp>
#include <cugip/math.hpp>
#include <cugip/image.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image.hpp>
#include <cugip/procedural_views.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using namespace cugip;

static const float cEpsilon = 0.00001;

BOOST_AUTO_TEST_CASE(CheckerBoardSum)
{
	auto view = cugip::checkerBoard<int, 2>(int(1), int(0), Int2(2, 2), Int2(16, 16));

	float sum = reduce(view, 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(sum, 16 * 16 / 2);
}

BOOST_AUTO_TEST_CASE(CopyPaddedHostImage)
{
	device_image<int, 3> deviceImage(3, 3, 3);
	host_image<int, 3> hostImage(4, 4, 3);

	auto hostView = view(hostImage);
	for (int k = 0; k < 3; ++k) {
		for (int j = 0; j < 4; ++j) {
			for (int i = 0; i < 4; ++i) {
				hostView[Int3(i, j, k)] = 1;
			}
		}
	}

	auto subimageView = makeConstHostImageView(hostImage.pointer(), Int3(3, 3, 3), hostImage.strides());
	copy(subimageView, view(deviceImage));
	auto flat = constantImage(1, Int3(3, 3, 3));
	int diff = reduce(square(subtract(const_view(deviceImage), flat)), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(diff, 0);
}

BOOST_AUTO_TEST_CASE(FlatSum)
{
	auto view1 = constantImage(3.0f, Int3(512, 512, 64));

	float sum = reduce(view1, 0.0f, thrust::plus<float>());
	BOOST_CHECK_CLOSE(sum, 3.0f * 512 * 512 * 64, cEpsilon);
}

// To test values of procedural device view we need to copy it to the memory based device view
// and then to the host memory view, where we can access the values easily.
BOOST_AUTO_TEST_CASE(FlatCopyAndCopyToHost)
{
	auto flat = constantImage(7.0f, Int3(16, 16, 2));
	device_image<int, 3> deviceImage(16, 16, 2);
	host_image<int, 3> hostImage(16, 16, 2);

	copy(flat, view(deviceImage));

	float sum = reduce(const_view(deviceImage), 0.0f, thrust::plus<float>());
	BOOST_CHECK_CLOSE(sum, 7.0f * elementCount(flat), cEpsilon);

	copy(const_view(deviceImage), view(hostImage));

	auto hostView = const_view(hostImage);
	for (int i = 0; i < elementCount(hostView); ++i) {
		BOOST_CHECK_CLOSE(7.0f, linear_access(hostView, i), cEpsilon);
	}
}

BOOST_AUTO_TEST_CASE(MultiViewOperator)
{
	device_image<int, 3> deviceImage(16, 16, 2);

	auto result = nAryOperator(SumValuesFunctor(), const_view(deviceImage), const_view(deviceImage));
	copy(result, view(deviceImage));
}
