#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

#include <cugip/image.hpp>
//#include <cugip/memory_view.hpp>
//#include <cugip/memory.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/view_arithmetics.hpp>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cugip/for_each.hpp>


using namespace cugip;

void relabeling(
	host_image_view<int, 3> aInput,
	int aStart)
{
	std::vector<int> usedLabels(elementCount(aInput) + 1);

	cugip::for_each(aInput, [&](int aValue) {
			usedLabels[aValue] = 1;
			return aValue;
		});
	thrust::inclusive_scan(usedLabels.begin(), usedLabels.end(), usedLabels.begin());
	cugip::for_each(aInput, [&](int aValue) {
			return usedLabels[aValue] + aStart;
		});
}
