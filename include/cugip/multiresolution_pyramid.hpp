#pragma once

#include <cugip/math.hpp>

namespace cugip {

template<int tDimension>
struct multiresolution_level {
	float scale;
	simple_vector<int, tDimension> resolution;
};

template<typename TImage>
class multiresolution_pyramid {
	using level_info = multiresolution_level<dimension<TImage>::value>;

	const level_info&
	level(int index) const {
		return mLevel[index];
	}

	std::vector<level_info> mLevel;
};

template<typename TImageView>
class multiresolution_pyramid_view {

};

template<typename TInputView, typename TPyramidView, typename TTmpView, typename TAlgorithm>
void compute_multiresolution_pyramid(TInputView input, TPyramidView pyramid, TTmpView tmpImage, TAlgorithm postprocess)
{
	static constexpr int cDimension = dimension<TInputView>::value;
	for (int i = 0; i < pyramid.level_count(); ++i) {
		auto scale = pyramid.level_info(i).scale;
		auto resolution = pyramid.level_info(i).resolution;
		auto tmpView = subview(tmpImage, simple_vector<int, cDimension>(), resolution);

		scale_image(input, tmpView, scale);

		auto outputLevel = pyramid.level(i);

		postprocess(tmpView, outputLevel);
	}
}

}  //namespace cugip
