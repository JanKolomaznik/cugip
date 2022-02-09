#pragma once

#include <boost/exception/all.hpp>

// #include <cugip/math/vector.hpp>
#include <cugip/region.hpp>
#include <filesystem>

namespace cugip {

using MessageErrorInfo = boost::error_info<struct tag_message, std::string>;

/// Error info containing file path.
using FilenameErrorInfo = boost::error_info<struct tag_filename, std::filesystem::path>;
using DirNameErrorInfo = boost::error_info<struct tag_dirname, std::filesystem::path>;

using Dimension1DErrorInfo = boost::error_info<struct tag_dimension_1d, int64_t>;
using Dimension2DErrorInfo = boost::error_info<struct tag_dimension_2d, vect2i_t>;
using Dimension3DErrorInfo = boost::error_info<struct tag_dimension_3d, vect3i_t>;
template<typename TView>
auto dimensionErrorInfo(const TView &aView) {
	if constexpr(TView::cDimension == 1) {
		return Dimension1DErrorInfo(aView.dimensions());
	}
	if constexpr(TView::cDimension == 2) {
		return Dimension2DErrorInfo(aView.dimensions());
	}
	if constexpr(TView::cDimension == 3) {
		return Dimension3DErrorInfo(aView.dimensions());
	}
}

// TODO - long coords
using SourceRegion1DErrorInfo = boost::error_info<struct tag_source_region_1d, region<1>>;
using SourceRegion2DErrorInfo = boost::error_info<struct tag_source_region_2d, region<2>>;
using SourceRegion3DErrorInfo = boost::error_info<struct tag_source_region_3d, region<3>>;

template<int tDim>
auto sourceRegionErrorInfo(const region<tDim> &aRegion) {
	if constexpr(tDim == 1) {
		return SourceRegion1DErrorInfo(aRegion);
	}
	if constexpr(tDim == 2) {
		return SourceRegion2DErrorInfo(aRegion);
	}
	if constexpr(tDim == 3) {
		return SourceRegion3DErrorInfo(aRegion);
	}
}

using TargetRegion1DErrorInfo = boost::error_info<struct tag_target_region_1d, region<1>>;
using TargetRegion2DErrorInfo = boost::error_info<struct tag_target_region_2d, region<2>>;
using TargetRegion3DErrorInfo = boost::error_info<struct tag_target_region_3d, region<3>>;

template<int tDim>
auto targetRegionErrorInfo(const region<tDim> &aRegion) {
	if constexpr(tDim == 1) {
		return TargetRegion1DErrorInfo(aRegion);
	}
	if constexpr(tDim == 2) {
		return TargetRegion2DErrorInfo(aRegion);
	}
	if constexpr(tDim == 3) {
		return TargetRegion3DErrorInfo(aRegion);
	}
}

// First the slice number, then slicing dimension
using InvalidSliceErrorInfo = boost::error_info<struct tag_invalid_slice, vect2i_t>;

} // namespace cugip

