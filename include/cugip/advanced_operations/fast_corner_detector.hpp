#pragma once

#include <cugip/math.hpp>
/*#include <cugip/host_image.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/image.hpp>
#include <cugip/image_view.hpp>*/

namespace cugip {

//static constexpr simple_vector<simple_vector<int, 2>, 4> cOffsets {
static constexpr vect2i_t cOffsets[4] {
	vect2i_t( 0, -3 ),
	vect2i_t( 1, -3 ),
	vect2i_t( 2, -2 ),
	vect2i_t( 3, -1 ),
};

//static constexpr simple_vector<simple_vector<int, 2>, 4>cQuadrants {
static constexpr vect2i_t cQuadrants[4] {
	vect2i_t(1, 1),
	vect2i_t(1, -1),
	vect2i_t(-1, -1),
	vect2i_t(-1, 1),
};

template<typename TValue>
int CompareToRange(TValue aLower, TValue aUpper, TValue aValue) {
	if (aValue < aLower) {
		return -1;
	}
	if (aValue > aUpper) {
		return 1;
	}
	return 0;
}

struct ComputeSalience
{
	template<typename TLocator>
	float operator()(TLocator locator)
	{
		simple_vector<int, 16> comparisons;
		auto centerValue = locator.get();
		auto upper = centerValue + mThreshold;
		auto lower = centerValue - mThreshold;
		float upperSaliency = 0.0f;
		float lowerSaliency = 0.0f;
		float upperCount = 0;
		float lowerCount = 0;

		for (int i = 0; i < 4; ++i) {  //TODO correct
			for (int j = 0; j < 4; ++j) {
				auto offset = product(cOffsets[j], cQuadrants[i]);
				auto testedValue = locator[offset];
				auto comparison = CompareToRange(lower, upper, testedValue);
				switch (comparison) {
				case -1:
					lowerSaliency += lower - testedValue;
					++lowerCount;
					break;
				case 1:
					upperSaliency += testedValue - upper;
					++upperCount;
					break;
				}

				comparisons[4*i + j] = comparison;
			}
		}

		if (max(lowerCount, upperCount) < 9) {
			return 0;
		}
		if (lowerCount > upperCount) {
			return lowerSaliency;
		} else {
			return upperSaliency;
		}
	}

	float mThreshold; // TODO - change type, make generic
};

template<typename TInput, typename TSaliency, typename TRunConfig>
void fast_corner_saliency(TInput aInput, TSaliency aSaliency, TRunConfig aConfig) {
	transform_locator(aInput, aSaliency, ComputeSalience{ 5 }); // TODO set threshold, run policy
}

} //namespace cugip
