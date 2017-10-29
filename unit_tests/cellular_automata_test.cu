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
