#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__


#include <cugip/advanced_operations/graph_cut.hpp>

void
test_graph_cut()
{

}

