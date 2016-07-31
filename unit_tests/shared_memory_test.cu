#define BOOST_TEST_MODULE SharedMemoryTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/cuda_utils.hpp>
#include <cugip/utils.hpp>
#include <cugip/tuple.hpp>
#include <cugip/neighborhood.hpp>
#include <cugip/access_utils.hpp>
#include <cugip/math/symmetric_tensor.hpp>
#include <cugip/math/eigen.hpp>
#include <cugip/math/matrix.hpp>
#include <cugip/detail/shared_memory.hpp>
#include <cugip/image.hpp>
#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/view_arithmetics.hpp>
#include <cugip/reduce.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>

using namespace cugip;


template<typename TInView, typename TOutView, typename TSize>
CUGIP_GLOBAL void
testIdentitySharedMemory(TInView aIn, TOutView aOut)
{
	__shared__ cugip::detail::SharedMemory<int, TSize> buffer;
	auto blockCoord = mapBlockIdxToViewCoordinates<dimension<TInView>::value>();
	auto extents = aIn.dimensions();
	auto coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView>::value>();

	buffer.load(aIn, blockCoord);

	__syncthreads();

	if (coord < extents) {
		aOut[coord] = buffer.get(currentThreadIndex());
		//printf("%d, %d, %d \n", coord[0], coord[1], coord[2]);
	}
}


BOOST_AUTO_TEST_CASE(IdentityWithPreload)
{
	device_image<int, 3> inImage(8, 8, 8);
	//device_image<int, 3> inImage(32, 32, 32);
	device_image<int, 3> outImage(inImage.dimensions());

	auto inView = view(inImage);
	//copy(UniqueIdDeviceImageView<3>(vect3i_t(32, 32, 32)), inView);
	copy(checkerBoard(2, 10, vect3i_t(3, 7, 5), inImage.dimensions()), inView);
	auto outView = view(outImage);
	dim3 blockSize(8, 8, 8);
	dim3 gridSize(5, 5, 5);
	typedef decltype(inView) InView;
	typedef decltype(outView) OutView;
	testIdentitySharedMemory<
		InView, 
		OutView,
		StaticSize<8, 8, 8> ><<<gridSize, blockSize>>>(inView, view(outImage));
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	std::cout << sum(inView) << "; " << sum(outView) << "\n";
	auto diff = sum(square(subtract(inView, const_view(outImage))));

	BOOST_CHECK_EQUAL(diff, 0);
}

BOOST_AUTO_TEST_CASE(IdentityWithPreload8LoadsPerThread)
{
	device_image<int, 3> inImage(8, 8, 8);
	//device_image<int, 3> inImage(32, 32, 32);
	device_image<int, 3> outImage(inImage.dimensions());

	auto inView = view(inImage);
	//copy(UniqueIdDeviceImageView<3>(vect3i_t(32, 32, 32)), inView);
	copy(checkerBoard(2, 10, vect3i_t(3, 7, 5), inImage.dimensions()), inView);
	auto outView = view(outImage);
	dim3 blockSize(4, 4, 4);
	dim3 gridSize(5, 5, 5);
	typedef decltype(inView) InView;
	typedef decltype(outView) OutView;
	testIdentitySharedMemory<
		InView, 
		OutView,
		StaticSize<8, 8, 8> ><<<gridSize, blockSize>>>(inView, view(outImage));
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());

	std::cout << sum(inView) << "; " << sum(outView) << "\n";
	auto diff = sum(square(subtract(inView, const_view(outImage))));

	BOOST_CHECK_EQUAL(diff, 0);
}
