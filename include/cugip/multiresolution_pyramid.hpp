#pragma once

#include <cugip/math.hpp>
#include <boost/rational.hpp>


namespace cugip {

class IncompatibleMultiresolutionPyramids: public ExceptionBase {};

typedef boost::error_info<struct tag_pyramid_level, int> PyramidLevelErrorInfo;

using scale_type = boost::rational<int>;

template<int tDimension>
struct multiresolution_level {
	scale_type scale;
	//float scale;
	simple_vector<int, tDimension> resolution;
};

template<int tDimension>
using multiresolution_levels = std::vector<multiresolution_level<tDimension>>;

template<typename TImage>
class multiresolution_pyramid_impl {
public:
	static constexpr int dimension = cugip::dimension<TImage>::value;
	using info = multiresolution_level<dimension>;
	using image = TImage;


	multiresolution_pyramid_impl() = default;
	multiresolution_pyramid_impl(multiresolution_levels<dimension> aLevels)
		: mLevelInfos(std::move(aLevels))
		, mLevel(mLevelInfos.size())
	{}

	int
	count() const {
		return mLevelInfos.size();
	}

	const info&
	level_info(int index) const {
		return mLevelInfos.at(index);
	}

	TImage&
	level(int index) {
		return mLevel.at(index);
	}

	const TImage&
	level(int index) const {
		return mLevel.at(index);
	}

	std::vector<info> mLevelInfos;
	std::vector<TImage> mLevel;
};

template<typename TImage>
class multiresolution_pyramid : public multiresolution_pyramid_impl<TImage>
{
public:
	static constexpr int dimension = cugip::dimension<TImage>::value;
	using image_view = decltype(cugip::view(std::declval<TImage>()));
	using const_image_view = decltype(cugip::const_view(std::declval<TImage>()));

	//using multiresolution_pyramid_impl<TImage>::multiresolution_pyramid_impl;
	multiresolution_pyramid() = default;
	multiresolution_pyramid(multiresolution_levels<dimension> aLevels)
		: multiresolution_pyramid_impl<TImage>(std::move(aLevels))
	{
		for (int i = 0; i < this->mLevel.size(); ++i) {
			this->mLevel[i] = TImage(this->mLevelInfos[i].resolution);
		}
	}


	multiresolution_pyramid(TImage &&aImage, multiresolution_levels<dimension> aLevels)
		: multiresolution_pyramid_impl<TImage>(std::move(aLevels))
	{
		for (int i = 1; i < this->mLevel.size(); ++i) {
			this->mLevel[i] = TImage(this->mLevelInfos[i].resolution);
		}
		this->mLevel[0] = std::move(aImage);
	}
};

template<typename TImageView>
class multiresolution_pyramid_view : public multiresolution_pyramid_impl<TImageView>
{
public:
	static_assert(is_image_view<TImageView>::value, "Template parameter must follow image view concept");

};

template<typename TImage>
auto view(const multiresolution_pyramid<TImage> &aPyramid)// -> multiresolution_pyramid_view<typename  multiresolution_pyramid<TImage>::image_view>
{
	//multiresolution_pyramid<decltype(view(aPyramid))> result;
	multiresolution_pyramid_view<typename  multiresolution_pyramid<TImage>::image_view> result;
	result.mLevelInfos = aPyramid.mLevelInfos;

	result.mLevel.resize(result.mLevelInfos.size());
	for (int i = 0; i < result.mLevel.size(); ++i) {
		result.mLevel[i] = view(aPyramid.level(i));
	}

	return result;
}

template<typename TImage>
auto const_view(const multiresolution_pyramid<TImage> &aPyramid)// -> multiresolution_pyramid_view<typename  multiresolution_pyramid<TImage>::const_image_view>
{
	 multiresolution_pyramid_view<typename  multiresolution_pyramid<TImage>::const_image_view> result;
	result.mLevelInfos = aPyramid.mLevelInfos;

	result.mLevel.resize(result.mLevelInfos.size());
	for (int i = 0; i < result.mLevel.size(); ++i) {
		result[i] = const_view(aPyramid.level(i));
	}

	return result;
}


template<typename TInputView, typename TPyramidView, typename TTmpView, typename TAlgorithm>
void compute_multiresolution_pyramid(TInputView input, TPyramidView pyramid, TTmpView tmpImage, TAlgorithm postprocess)
{
	/*static constexpr int cDimension = dimension<TInputView>::value;
	for (int i = 0; i < pyramid.level_count(); ++i) {
		auto scale = pyramid.level_info(i).scale;
		auto resolution = pyramid.level_info(i).resolution;
		auto tmpView = subview(tmpImage, simple_vector<int, cDimension>(), resolution);

		scale_image(input, tmpView, scale);

		auto outputLevel = pyramid.level(i);

		postprocess(tmpView, outputLevel);
	}*/
}

template<typename TPyramidView, typename TTmpView, typename TAlgorithm>
void compute_multiresolution_pyramid_sublevels(TPyramidView pyramid, TTmpView tmpImage, TAlgorithm postprocess)
{
	/*static constexpr int cDimension = dimension<TInputView>::value;
	for (int i = 1; i < pyramid.level_count(); ++i) {
		auto scale = pyramid.level_info(i).scale;
		auto resolution = pyramid.level_info(i).resolution;
		auto tmpView = subview(tmpImage, simple_vector<int, cDimension>(), resolution);

		scale_image(input, tmpView, scale);

		auto outputLevel = pyramid.level(i);

		postprocess(tmpView, outputLevel);
	}*/
}

template<typename TPyramid1, typename TPyramid2, typename TFunctor>
void
apply_to_all_levels(TPyramid1 aPyramid1, TPyramid2 aPyramid2, TFunctor aFunctor)
{
	if (aPyramid1.count() != aPyramid2.count()) {
		CUGIP_THROW(IncompatibleMultiresolutionPyramids());
	}

	for (int i = 0; i < aPyramid1.count(); ++i) {
		try {
			aFunctor(aPyramid1.level(i), aPyramid2.level(i));
		} catch (ExceptionBase &e) {
			e << PyramidLevelErrorInfo(i);
		}
	}
}


template<typename TPyramidView>
void dump_pyramid(TPyramidView aPyramid, boost::filesystem::path aDirectory)
{
	if (!boost::filesystem::create_directories(aDirectory)) {
		//CUGIP_THROW(DirectoryNotCreated() << DirNameErrorInfo(aDirectory));
	}
	for (int i = 0; i < aPyramid.count(); ++i) {
		std::string prefix = "level_" + std::to_string(i) + "_";
		dump_view(aPyramid.level(i), aDirectory / prefix);
	}
}


}  //namespace cugip
