#define BOOST_TEST_MODULE UtilsTest
#include <boost/test/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

/*CUGIP_GLOBAL void
testBlockScanIn(int *output)
{
	__shared__ int buffer[512 + 1];

	auto sum = cugip::block_prefix_sum_in<int>(threadIdx.x, 512, threadIdx.x, buffer);
	if (threadIdx.x == 0) {
		printf("Total %d\n", sum.total);
	}
	__syncthreads();
	buffer[threadIdx.x] = - 1;
	output[threadIdx.x] = sum.current - (threadIdx.x + 1) * threadIdx.x / 2;
}



CUGIP_GLOBAL void
testBlockScanEx(int *output)
{
	__shared__ int buffer[512 + 1];

	auto sum = cugip::block_prefix_sum_ex<int>(threadIdx.x, 512, threadIdx.x, buffer);
	if (threadIdx.x == 0) {
		printf("Total %d\n", sum.total);
	}
	__syncthreads();
	buffer[threadIdx.x] = - 1;
	output[threadIdx.x] = sum.current - (threadIdx.x + 1) * threadIdx.x / 2 + threadIdx.x;
}

BOOST_AUTO_TEST_CASE(blockScanIn)
{

	thrust::device_vector<int> buffer(512);

	testBlockScanIn<<<1, 512>>>(thrust::raw_pointer_cast(buffer.data()));
	cudaThreadSynchronize();

	int result = thrust::reduce(buffer.begin(), buffer.end(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(result, 0);
}

BOOST_AUTO_TEST_CASE(blockScanEx)
{

	thrust::device_vector<int> buffer(512);

	testBlockScanEx<<<1, 512>>>(thrust::raw_pointer_cast(buffer.data()));
	cudaThreadSynchronize();

	int result = thrust::reduce(buffer.begin(), buffer.end(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(result, 0);
}*/
