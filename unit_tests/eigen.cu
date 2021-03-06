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
	symmetric_tensor<float, 3> m1{1, 0, 0, 1, 0, 1};

	auto ev = eigen_values(m1);
	BOOST_CHECK_EQUAL(tuple1.get<0>(), 3);
	BOOST_CHECK_EQUAL(tuple1.get<1>(), 1.5f);
	BOOST_CHECK_EQUAL(tuple1.get<2>(), true);

	BOOST_CHECK_EQUAL(get<0>(tuple1), 3);
	BOOST_CHECK_EQUAL(get<1>(tuple1), 1.5f);
	BOOST_CHECK_EQUAL(get<2>(tuple1), true);

	auto tuple2 = tuple1;
	BOOST_CHECK_EQUAL(get<0>(tuple2), 3);
	BOOST_CHECK_EQUAL(get<1>(tuple2), 1.5f);
	BOOST_CHECK_EQUAL(get<2>(tuple2), true);

	Tuple<int, float, bool> tuple3;
	tuple3 = tuple1;
	BOOST_CHECK_EQUAL(get<0>(tuple3), 3);
	BOOST_CHECK_EQUAL(get<1>(tuple3), 1.5f);
	BOOST_CHECK_EQUAL(get<2>(tuple3), true);
}

struct InitTuple
{
	CUGIP_DECL_HYBRID
	void operator()(Tuple<int, float, bool> &t) {
		t = Tuple<int, float, bool>(3, 1.5f, true);
	}
};

struct ModifyTuple
{
	CUGIP_DECL_HYBRID
	void operator()(Tuple<int, float, bool> &t) {
		t = Tuple<int, float, bool>(3, 1.5f, true);
	}
};

BOOST_AUTO_TEST_CASE(TupleAccessDevice)
{
	thrust::device_vector<Tuple<int, float, bool>> tuples1(10);

	thrust::for_each(tuples1.begin(), tuples1.end(), InitTuple());

	thrust::host_vector<Tuple<int, float, bool>> tuples2;
	tuples2 = tuples1;

	for (int i = 0; i < tuples2.size(); ++i) {
		BOOST_CHECK_EQUAL(get<0>(tuples2[i]), 3);
		BOOST_CHECK_EQUAL(get<1>(tuples2[i]), 1.5f);
		BOOST_CHECK_EQUAL(get<2>(tuples2[i]), true);
	}
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

BOOST_AUTO_TEST_CASE(VonNeumannNeighborhood2D)
{
	using namespace cugip;
	VonNeumannNeighborhood<2> neighborhood;
	for (int i = 0; i < 5; ++i) {
		BOOST_CHECK_EQUAL(neighborhood.offset(i), neighborhood.offset2(i));
	}
}

BOOST_AUTO_TEST_CASE(VonNeumannNeighborhood3D)
{
	using namespace cugip;
	VonNeumannNeighborhood<3> neighborhood;
	for (int i = 0; i < 7; ++i) {
		BOOST_CHECK_EQUAL(neighborhood.offset(i), neighborhood.offset2(i));
	}
}

BOOST_AUTO_TEST_CASE(MooreNeighborhood3D)
{
	using namespace cugip;
	MooreNeighborhood<3> neighborhood;
	for (int i = 0; i < 27; ++i) {
		BOOST_CHECK_EQUAL(neighborhood.offset(i), neighborhood.offset2(i));
		//std::cout << neighborhood.offset(i) << neighborhood.offset2(i) << std::endl;
	}
}

BOOST_AUTO_TEST_CASE(BlockedOrderAccessIndex)
{
	using namespace cugip;
	BOOST_CHECK_EQUAL(0, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(0, 0, 0)));
	BOOST_CHECK_EQUAL(7, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(1, 1, 1)));
	BOOST_CHECK_EQUAL(57, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(1, 1, 3)));
	BOOST_CHECK_EQUAL(8, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(2, 0, 0)));
	BOOST_CHECK_EQUAL(28, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(2, 2, 0)));
	BOOST_CHECK_EQUAL(78, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(2, 2, 2)));
	BOOST_CHECK_EQUAL(124, get_blocked_order_access_index<2>(Int3(5,5,5), Int3(4, 4, 4)));

	BOOST_CHECK_EQUAL(0, get_blocked_order_access_index<3>(Int3(5,5,5), Int3(0, 0, 0)));
	BOOST_CHECK_EQUAL(26, get_blocked_order_access_index<3>(Int3(5,5,5), Int3(2, 2, 2)));
	BOOST_CHECK_EQUAL(124, get_blocked_order_access_index<3>(Int3(5,5,5), Int3(4, 4, 4)));
}
