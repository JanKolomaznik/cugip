#define BOOST_TEST_MODULE EigenTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/math/eigen.hpp>

using namespace cugip;

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

*/


BOOST_AUTO_TEST_CASE(TupleAccess)
{
	/*symmetric_tensor<float, 3> m1{1, 0, 0, 1, 0, 1};

	auto ev = eigen_values(m1);*/
}


BOOST_AUTO_TEST_CASE(SymmetricTensorAccess)
{
	using namespace cugip;
	symmetric_tensor<int, 2> m1;
	for (int i = 0; i < 3; ++i) {
		m1[i] = i + 1;
	}

	symmetric_tensor<int, 4> m2;
	for (int i = 0; i < 10; ++i) {
		m2[i] = i + 1;
	}

	BOOST_CHECK_EQUAL((get<0, 0>(m1)), 1);
	BOOST_CHECK_EQUAL((get<0, 1>(m1)), 2);
	BOOST_CHECK_EQUAL((get<1, 1>(m1)), 3);

	BOOST_CHECK_EQUAL((get<0, 3>(m2)), 4);
	BOOST_CHECK_EQUAL((get<1, 1>(m2)), 5);
	BOOST_CHECK_EQUAL((get<2, 2>(m2)), 8);
	BOOST_CHECK_EQUAL((get<3, 3>(m2)), 10);
	BOOST_CHECK_EQUAL((get<2, 3>(m2)), 9);
	BOOST_CHECK_EQUAL((get<3, 2>(m2)), 9);
}

BOOST_AUTO_TEST_CASE(SymmetricTensor3x3EigenValues)
{
	using namespace cugip;
	symmetric_tensor<float, 3> tensor;

	tensor[0] = 1.0f;
	tensor[1] = 0.0f;
	tensor[2] = 0.0f;
	tensor[3] = 2.0f;
	tensor[4] = 0.0f;
	tensor[5] = 3.0f;

	std::cout << eigen_values(tensor) << "\n";
	/*std::cout << eigen_vector<0, 1>(3.0f, tensor) << "\n";
	std::cout << eigen_vector<0, 2>(2.0f, tensor) << "\n";
	std::cout << eigen_vector<1, 2>(1.0f, tensor) << "\n";*/
	std::cout << eigen_vectors(tensor, eigen_values(tensor)) << "\n";

}
