#if defined(__CUDACC__)
#ifndef BOOST_NOINLINE
#	define BOOST_NOINLINE __attribute__ ((noinline))
#endif //BOOST_NOINLINE
#endif //__CUDACC__

//#include "itkImage.h"

#include <cugip/image.hpp>
#include <cugip/advanced_operations/nonlocal_means.hpp>
#include <cugip/memory_view.hpp>
#include <cugip/memory.hpp>
#include <cugip/copy.hpp>
#include <cugip/host_image_view.hpp>


#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <functional>
#include <utility>

//typedef itk::Image<float, 3> ImageType;
using namespace cugip;




void
denoise(float *aInput, float *aOutput, size_t aWidth, size_t aHeight, size_t aDepth, float aVariance, int aPatchRadius, int aSearchRadius)
//denoise(ImageType::Pointer aInput, ImageType::Pointer aOutput)
{
	//cugip::const_host_memory_3d<float> pom(aInput, aWidth, aHeight, aDepth, aWidth * sizeof(float));

	//cugip::const_memory_view<float, 3> inView(cugip::const_host_memory_3d<float>(aInput, aWidth, aHeight, aDepth, aWidth * sizeof(float)));
	//cugip::memory_view<float, 3> outView(cugip::host_memory_3d<float>(aOutput, aWidth, aHeight, aDepth, aWidth * sizeof(float)));
	auto inView = makeConstHostImageView(aInput, vect3i_t(aWidth, aHeight, aDepth));
	auto outView = makeHostImageView(aOutput, vect3i_t(aWidth, aHeight, aDepth));

	D_PRINT(aDepth);
	D_PRINT(inView.dimensions());

	//cugip::print_error_enums();
	D_PRINT(cugip::cudaMemoryInfoText());
	cugip::device_image<float, 3> inImage(inView.dimensions());
	cugip::device_image<float, 3> outImage(inView.dimensions());
	D_PRINT(cugip::cudaMemoryInfoText());

	cugip::copy(inView, cugip::view(inImage));

	D_PRINT("nonlocal_means ...");
	D_PRINT("variance = " << aVariance);
	D_PRINT("patch radius = " << aPatchRadius);
	D_PRINT("search radius = " << aSearchRadius);

	using Tuple = std::tuple<int, int>;
	using InView = decltype(cugip::const_view(inImage));
	using OutView = decltype(cugip::view(outImage));

	static std::unordered_map<Tuple, std::function<void(InView, OutView, float)>, boost::hash<Tuple>> implementations = {
		{ Tuple(1, 2), [](InView aInView, OutView aOutView, float aVariance) { nonlocal_means(aInView, aOutView, nl_means_parameters<1, 2>(aVariance)); }},
		{ Tuple(1, 3), [](InView aInView, OutView aOutView, float aVariance) { nonlocal_means(aInView, aOutView, nl_means_parameters<1, 3>(aVariance)); }},
		{ Tuple(1, 4), [](InView aInView, OutView aOutView, float aVariance) { nonlocal_means(aInView, aOutView, nl_means_parameters<1, 4>(aVariance)); }},
		{ Tuple(1, 5), [](InView aInView, OutView aOutView, float aVariance) { nonlocal_means(aInView, aOutView, nl_means_parameters<1, 5>(aVariance)); }},

		{ Tuple(2, 2), [](InView aInView, OutView aOutView, float aVariance) { nonlocal_means(aInView, aOutView, nl_means_parameters<2, 2>(aVariance)); }},
		{ Tuple(2, 3), [](InView aInView, OutView aOutView, float aVariance) { nonlocal_means(aInView, aOutView, nl_means_parameters<2, 3>(aVariance)); }},
		{ Tuple(2, 4), [](InView aInView, OutView aOutView, float aVariance) { nonlocal_means(aInView, aOutView, nl_means_parameters<2, 4>(aVariance)); }},
	};

	auto search = implementations.find(Tuple(aPatchRadius, aSearchRadius));
   	if(search != implementations.end()) {
		search->second(cugip::const_view(inImage), cugip::view(outImage), aVariance);
	} else {
		throw std::runtime_error("Configuration not handled!");
	}

	//cugip::nonlocal_means(cugip::const_view(inImage), cugip::view(outImage), cugip::nl_means_parameters<2, 4>(aVariance));

	cugip::copy(cugip::view(outImage), outView);
	D_PRINT("nonlocal_means done!");

	CUGIP_CHECK_ERROR_STATE("denoise");

}
