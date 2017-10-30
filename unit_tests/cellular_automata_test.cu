#define BOOST_TEST_MODULE CellularAutomataTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cuda.h>

#include <cugip/gil_utils.hpp>
#include <cugip/utils.hpp>
#include <cugip/reduce.hpp>
#include <cugip/math.hpp>
#include <cugip/image.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image.hpp>
#include <cugip/procedural_views.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cugip/cellular_automata/cellular_automata.hpp>

#include <boost/gil/gil_all.hpp>
using namespace cugip;

//static const float cEpsilon = 0.00001;


BOOST_AUTO_TEST_CASE(Conway)
{
        //boost::gil::gray8_image_t gray_out(50, 50);
        //device_image<int, 2> im(50,50);
        //copy(boost::gil::const_view(gray_out), view(im));
	//CellularAutomaton<MooreNeighborhood<2>, ConwayRule> automaton;
	//automaton.iterate(1);
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
