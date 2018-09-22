#define BOOST_TEST_MODULE ScanTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/reduce.hpp>
#include <cugip/math.hpp>
#include <cugip/image.hpp>
#include <cugip/image_dumping.hpp>
#include <cugip/host_image.hpp>
#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/scan.hpp>
//#include <cugip/transform.hpp>

using namespace cugip;

static const float cEpsilon = 0.00001;


/*BOOST_AUTO_TEST_CASE(PrefixSumEmptyConstantImage)
{
	device_image<int, 2> deviceImage(128, 128);
	device_image<int, 2> tmpImage(128, 128);

	auto input = constantImage(0, deviceImage.dimensions());

	scan(input, view(deviceImage), view(tmpImage), 0, IntValue<0>{}, SumValuesFunctor{});
	
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());


	auto difference = sum_differences(const_view(deviceImage), input, int{0});

	BOOST_CHECK_EQUAL(difference, 0);
}*/

BOOST_AUTO_TEST_CASE(PrefixSumConstantImage)
{
	device_image<int, 2> deviceImage(128, 128);
	device_image<int, 2> tmpImage(deviceImage.dimensions());

	auto input = constantImage(1, deviceImage.dimensions());

	scan(input, view(deviceImage), view(tmpImage), 0, IntValue<0>{}, SumValuesFunctor{});
	
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());


	auto groundTruth = MeshGridView<2>(vect2i_t(), input.dimensions(), 0);
	//static_assert(is_image_view<decltype(const_view(deviceImage))>::value, "AAAA");
	//using T = std::enable_if<is_image_view<decltype(const_view(deviceImage))>::value>::type;
	auto difference = sum_differences(const_view(deviceImage), groundTruth, int{0});

	/*host_image<int, 2> hostImage(deviceImage.dimensions());
	copy(view(deviceImage), view(hostImage));
	print_view(view(hostImage));*/

	BOOST_CHECK_EQUAL(difference, 0);
}

BOOST_AUTO_TEST_CASE(PrefixSumConstantImageLarge)
{
	device_image<int, 2> deviceImage(2025, 256);
	device_image<int, 2> tmpImage(deviceImage.dimensions());

	auto input = constantImage(1, deviceImage.dimensions());

	scan(input, view(deviceImage), view(tmpImage), 0, IntValue<0>{}, SumValuesFunctor{});
	
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());


	auto groundTruth = MeshGridView<2>(vect2i_t(), input.dimensions(), 0);
	//static_assert(is_image_view<decltype(const_view(deviceImage))>::value, "AAAA");
	//using T = std::enable_if<is_image_view<decltype(const_view(deviceImage))>::value>::type;
	auto difference = sum_differences(const_view(deviceImage), groundTruth, int{0});

	//device_image<int, 2> deviceImage(2000, 1);
	/*host_image<int, 2> hostImage(deviceImage.dimensions());
	copy(view(deviceImage), view(hostImage));
	print_view(view(hostImage));*/

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
