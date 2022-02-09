#pragma once

#include <cugip/math/vector.hpp>
#include <cugip/transform.hpp>

namespace cugip {

template<typename TType>
class IntensityAlpha: public simple_vector<TType, 2> {};

template<typename TType>
class RGB: public simple_vector<TType, 3> {
	using simple_vector<TType, 3>::simple_vector;
};

template<typename TType>
class RGBA: public simple_vector<TType, 4> {};

template<typename TType>
using Intensity = TType;

using Intensity_8 = Intensity<uint8_t>;
using Intensity_16 = Intensity<uint16_t>;
using Intensity_32 = Intensity<uint32_t>;
using Intensity_32f = Intensity<float>;
using Intensity_64f = Intensity<double>;

using IntensityAlpha_8 = IntensityAlpha<uint8_t>;
using IntensityAlpha_16 = IntensityAlpha<uint16_t>;
using IntensityAlpha_32 = IntensityAlpha<uint32_t>;
using IntensityAlpha_32f = IntensityAlpha<float>;
using IntensityAlpha_64f = IntensityAlpha<double>;

using RGB_8 = RGB<uint8_t>;
using RGB_16 = RGB<uint16_t>;
using RGB_32 = RGB<uint32_t>;
using RGB_32f = RGB<float>;
using RGB_64f = RGB<double>;

using RGBA_8 = RGBA<uint8_t>;
using RGBA_16 = RGBA<uint16_t>;
using RGBA_32 = RGBA<uint32_t>;
using RGBA_32f = RGBA<float>;
using RGBA_64f = RGBA<double>;

struct RGBtoIntensityFtor {
	// proper constexpr
	CUGIP_DECL_HYBRID
	constexpr vect3f_t RGBWeights() const {
		constexpr vect3f_t cRGBWeights(0.2990f, 0.5870f, 0.1140f);

		return cRGBWeights;
	}

	template<typename TPrecision>
	CUGIP_DECL_HYBRID
	Intensity<TPrecision>
	operator()(const RGB<TPrecision> &aColor) const {
		// TODO - conversion
		return { TPrecision(sum(product(aColor, RGBWeights()))) };
	}
};


template<typename TRGBImage, typename TIntensityImage>
void
rgb_to_intesity(TRGBImage aIn, TIntensityImage aOut) {
	transform(aIn, aOut, RGBtoIntensityFtor{});
}

template<typename TChannel1, typename TChannel2>
inline CUGIP_DECL_HYBRID
auto operator+(const RGB<TChannel1> &aArg1, const RGB<TChannel2> &aArg2)
{
	RGB<decltype(TChannel1{} + TChannel2{})> res(aArg1);
	return res += aArg2;
}

template<typename TFactor, typename TChannel>
inline CUGIP_DECL_HYBRID
auto operator*(TFactor aFactor, RGB<TChannel> aArg)
{
	RGB<decltype(aFactor * TChannel{})> res;
	for (int i = 0; i < 3; ++i) {
		res[i] = aFactor * aArg[i];
	}
	return res;
}




} // namespace cugip
