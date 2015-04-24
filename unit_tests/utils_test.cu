#define BOOST_TEST_MODULE UtilsTest
#include <boost/test/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>


CUGIP_GLOBAL void
testBlockScan(int *output)
{
	__shared__ int buffer[512 + 1];

	int sum = cugip::block_prefix_sum_in<int>(threadIdx.x, 512, threadIdx.x, buffer);
	__syncthreads();
	output[threadIdx.x] = sum - (threadIdx.x + 1) * threadIdx.x / 2;
}


BOOST_AUTO_TEST_CASE(my_test)
{

	thrust::device_vector<int> buffer(512);

	testBlockScan<<<1, 512>>>(thrust::raw_pointer_cast(buffer.data()));
	cudaThreadSynchronize();

	int result = thrust::reduce(buffer.begin(), buffer.end(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(result, 0);
}
