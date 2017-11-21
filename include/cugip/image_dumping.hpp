#pragma once

#include <fstream>
#include <boost/filesystem.hpp>
#include <cugip/access_utils.hpp>

namespace cugip {

template<typename TView>
void dump_view_to_file(TView aView, boost::filesystem::path aFile)
{
	static_assert(is_image_view<TView>::value, "Works only on image views");
	static_assert(is_host_view<TView>::value, "Dump to file works only for host views");
	std::ofstream out;
	out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	out.open(aFile.string(), std::ofstream::out | std::ofstream::binary);

	for (int i = 0; i < elementCount(aView); ++i) {
		auto element = linear_access(aView, i);
		out.write(reinterpret_cast<const char *>(&element), sizeof(element));
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
void dump_view(TView aView, boost::filesystem::path aPrefix)
{
	boost::filesystem::path filename = aPrefix.string() + ("_" + dimensionsToString(aView.dimensions()) + ".raw");

	dump_view_to_file(aView, filename);
}


} // namespace cugip
