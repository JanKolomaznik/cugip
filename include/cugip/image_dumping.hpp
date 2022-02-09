#pragma once

#include <fstream>
#include <filesystem>
#include <cugip/access_utils.hpp>
#include <cugip/image_traits.hpp>
#include <cugip/copy.hpp>
#include <cugip/for_each.hpp>

namespace cugip {

class DirectoryNotCreated: public ExceptionBase {};

template<typename TView>
void dump_view_to_file_impl(std::filesystem::path aFile, TView aView)
{
	static_assert(is_image_view<TView>::value, "Works only on image views");
	static_assert(is_host_view<TView>::value, "Dump to file works only for host views");
	std::ofstream out;
	out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	out.open(aFile.string(), std::ofstream::out | std::ofstream::binary);

	for (int64_t i = 0; i < elementCount(aView); ++i) {
		auto element = linear_access(aView, i);
		out.write(reinterpret_cast<const char *>(&element), sizeof(element));
	}
}

template<typename TView>
void dump_view_to_file(TView aView, std::filesystem::path aFile)
{
	if constexpr(!is_host_view<TView>::value) {
		using TmpImage = typename cugip::image_view_traits<TView>::host_image_t;
		TmpImage tmpImage(aView.dimensions());
		cugip::copy(aView, cugip::view(tmpImage));
		dump_view_to_file_impl(aFile, cugip::const_view(tmpImage));
	} else {
		dump_view_to_file_impl(aFile, aView);
	}
}

template<int tDimension>
std::string
dimensionsToString(const simple_vector<int, tDimension> &aDimensions)
{
	std::string output = std::to_string(aDimensions[0]);
	for (int i = 1; i < tDimension; ++i) {
		output += 'x' + std::to_string(aDimensions[i]);
	}
	return output;
}

template<typename TView>
void dump_view(TView aView, std::filesystem::path aPrefix)
{
	std::filesystem::path filename = aPrefix.string() + ("_" + dimensionsToString(aView.dimensions()) + ".raw");

	dump_view_to_file(aView, filename);
}

template<typename TView>
void print_view(TView aView)
{
	static_assert(is_host_view<TView>::value, "Print works only for host views");
	for_each(aView, [](const int &aValue) { std::cout << aValue << " "; });
	std::cout << "\n";
}


} // namespace cugip
