#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>
#include <cugip/element.hpp>
#include <cugip/tuple.hpp>

namespace cugip {

template<typename TType>
struct negate
{
	CUGIP_DECL_HYBRID TType
	operator()(const TType &aArg)const
	{
		TType tmp;
		tmp.data[0] = 255 - aArg.data[0];
		tmp.data[1] = 255 - aArg.data[1];
		tmp.data[2] = 255 - aArg.data[2];
		return tmp;
	}
};

struct mandelbrot_ftor
{
	mandelbrot_ftor(
			dim_traits<2>::extents_t aExtents = dim_traits<2>::extents_t(),
			intervalf_t aXInterval = intervalf_t(-2.5f, 1.0f),
			intervalf_t aYInterval = intervalf_t(-1.0f, 1.0f)
			): extents(aExtents), xInterval(aXInterval), yInterval(aYInterval)
	{}

	CUGIP_DECL_HYBRID element_rgb8_t
	operator()(const element_rgb8_t &aArg, dim_traits<2>::coord_t aCoordinates)const
	{
		float xSize = xInterval.get<1>() - xInterval.get<0>();
		float ySize = yInterval.get<1>() - yInterval.get<0>();
		float x0 = (float(aCoordinates.get<0>()) / extents.get<0>()) * xSize + xInterval.get<0>();
		float y0 = (float(aCoordinates.get<1>()) / extents.get<1>()) * ySize + yInterval.get<0>();
		float x = 0.0f;
		float y = 0.0f;

		int iteration = 0;
		int max_iteration = 1000;

		while ( (x*x + y*y) < 2*2  &&  (iteration < max_iteration) )
		{
			float xtemp = x*x - y*y + x0;
			y = 2*x*y + y0;
			x = xtemp;

			++iteration;
		}
		element_rgb8_t tmp;
		tmp.data[0] = iteration % 256;
		tmp.data[1] = (iteration * 7) % 256;
		tmp.data[2] = (iteration * 13) % 256;
		return tmp;
	}
	dim_traits<2>::extents_t extents;
	intervalf_t xInterval;
	intervalf_t yInterval;
};

struct grayscale_ftor
{
	CUGIP_DECL_HYBRID element_gray8_t
	operator()(const element_rgb8_t &aArg)const
	{
		unsigned tmp = 0;
		tmp += aArg.data[0];
		tmp += aArg.data[1]*2;
		tmp += aArg.data[2];
		return tmp / 4;
	}
};

template<typename TInValue, typename TOutValue>
struct thresholding_ftor
{
	thresholding_ftor(TInValue aThreshold, TOutValue aUnderValue, TOutValue aUpperValue)
		: mThreshold(aThreshold), mUnderValue(aUnderValue), mUpperValue(aUpperValue)
	{ /*empty*/ }

	CUGIP_DECL_HYBRID TOutValue
	operator()(const TInValue &aArg)const
	{
		return (aArg < mThreshold) ? mUnderValue : mUpperValue;
	}
	TInValue mThreshold;
	TOutValue mUnderValue;
	TOutValue mUpperValue;
};

template<typename TInputType, typename TOutputType>
struct gradient_magnitude
{

	template<typename TAccessor>
	CUGIP_DECL_HYBRID TOutputType
	operator()(TAccessor aAccessor) const
	{
		return abs(aAccessor[typename TAccessor::diff_t(-1,0)] - aAccessor[typename TAccessor::diff_t()])
		     + abs(aAccessor[typename TAccessor::diff_t(0,-1)] - aAccessor[typename TAccessor::diff_t()]);
	}

};

struct convert_float_and_byte
{
	CUGIP_DECL_HYBRID unsigned char
	operator()(const float &aArg)const
	{
		return static_cast<unsigned char>(max(0.0f, min(255.0f, aArg)));
	}

	CUGIP_DECL_HYBRID float
	operator()(const unsigned char &aArg)const
	{
		return aArg;
	}
};


struct assign_color_ftor
{
	template<typename TInput>
	CUGIP_DECL_HYBRID element_rgb8_t
	operator()(const TInput &aArg)const
	{
		element_rgb8_t tmp;
		tmp.data[0] = (aArg * 123) % 256;
		tmp.data[1] = (aArg * 7) % 256;
		tmp.data[2] = (aArg * 13) % 256;
		return tmp;
	}
};

struct generate_basins_ftor
{
	CUGIP_DECL_HYBRID element_gray8_t
	operator()(const element_gray8_t &aArg, dim_traits<2>::coord_t aCoordinates)const
	{
		return max(abs((aCoordinates[0] % 500) - 250), abs((aCoordinates[1] % 500) - 250));
	}
};

struct generate_spiral_ftor
{
	generate_spiral_ftor(dim_traits<2>::extents_t aExtents)
		: extents(aExtents)
	{}

	CUGIP_DECL_HYBRID element_gray8_t
	operator()(const element_gray8_t &aArg, dim_traits<2>::coord_t aCoordinates)const
	{
		bool hit1 = false;
		// TODO cleanup
		int remainder = (extents[0] + 1) % 1;
		if (aCoordinates[0] < extents[0] >> 1) {
			if (aCoordinates[0] % 2 == 1) {
				hit1 = aCoordinates[1] > aCoordinates[0] && aCoordinates[1] <= (extents[0] - aCoordinates[0]);
			}
		} else {
			if (aCoordinates[0] % 2 == 1) {
				hit1 = aCoordinates[1] >= (extents[0] - aCoordinates[0] - 1) && aCoordinates[1] <= (aCoordinates[0]);
			}
		}
		bool hit2 = false;
		if (aCoordinates[1] < (extents[1] >> 1)) {
			if (aCoordinates[1] % 2 == 0) {
				hit2 = aCoordinates[0] >= (aCoordinates[1] - 1) && aCoordinates[0] < (extents[1] - aCoordinates[1]);
			}
		}  else {
			if (aCoordinates[1] % 2 == 1) {
				hit2 = aCoordinates[0] > (extents[1] - aCoordinates[1]) && aCoordinates[0] < (aCoordinates[1]);
			}
		}
		return (hit1 || hit2) ? 255 : 0;
	}
	dim_traits<2>::extents_t extents;
};

/// Usable in device/host code.
struct SquareFunctor {
	template<typename T>
	CUGIP_DECL_HYBRID
	auto operator()(T value) const -> decltype(value * value) {
		return value * value;
	}
};


/// Usable in device/host code.
struct SquareRootFunctor {
	template<typename T>
	CUGIP_DECL_HYBRID
	auto operator()(T value) const -> decltype(sqrt(value)) {
		return sqrt(value);
	}
};


/// Multiplies passed value by factor specified in constructor.
/// Usable in device/host code.
template<typename TFactor>
struct MultiplyByFactorFunctor {
	explicit MultiplyByFactorFunctor(TFactor factor) :
		factor_(factor)
	{}

	template<typename T>
	CUGIP_DECL_HYBRID
	auto operator()(T value) const -> decltype(std::declval<TFactor>() * value) {
		return factor_ * value;
	}

	TFactor factor_;
};

/// Adds specified constant to the passed value
/// Usable in device/host code.
template<typename TType>
struct AddValueFunctor {
	explicit AddValueFunctor(TType value) :
		value_(value)
	{}

	template<typename T>
	CUGIP_DECL_HYBRID
	auto operator()(T in_value) const -> decltype(std::declval<TType>() + in_value) {
		return value_ + in_value;
	}

	TType value_;
};

struct SumValuesFunctor {
	template<typename T1, typename T2>
	CUGIP_DECL_HYBRID
	auto operator()(T1 value1, T2 value2) const -> decltype(std::declval<T1>() + std::declval<T2>()) {
		return value1 + value2;
	}
};

struct ChannelSumFunctor {

	template<typename T>
	using remove_reference = typename std::remove_reference<T>::type;

	template<typename TValue>
	CUGIP_DECL_HYBRID
	auto operator()(TValue value) const -> remove_reference<decltype(value[0])> {
		auto result = value[0];
		for (int i = 1; i < dimension<TValue>::value; ++i) {
			result += value[i];
		}
		return result;
	}
};


/// Returns maximum of passed value and specified limit value
/// Usable in device/host code.
template<typename TType>
struct MaxFunctor {
	explicit MaxFunctor(TType limit) :
		limit_(limit)
	{}

	CUGIP_DECL_HYBRID
	TType operator()(TType in_value) const {
		return max(limit_, in_value);
	}

	TType limit_;
};

/// Returns minumum of passed value and specified limit value
/// Usable in device/host code.
template<typename TType>
struct MinFunctor {
	explicit MinFunctor(TType limit) :
		limit_(limit)
	{}

	CUGIP_DECL_HYBRID
	TType operator()(TType in_value) const {
		return min(limit_, in_value);
	}

	TType limit_;
};


/// If passed value is smaller than the limit return replacement value
/// Usable in device/host code.
template<typename TType>
struct LowerLimitFunctor {
	LowerLimitFunctor(TType limit, TType replacement) :
		limit_(limit),
		replacement_(replacement)
	{}

	CUGIP_DECL_HYBRID
	TType operator()(TType in_value) const {
		return in_value < limit_ ? replacement_ : in_value;
	}

	TType limit_;
	TType replacement_;
};

/// If passed value is bigger than the limit return replacement value
/// Usable in device/host code.
template<typename TType>
struct UpperLimitFunctor {
	UpperLimitFunctor(TType limit, TType replacement) :
		limit_(limit),
		replacement_(replacement)
	{}

	CUGIP_DECL_HYBRID
	TType operator()(TType in_value) const {
		return in_value > limit_ ? replacement_ : in_value;
	}

	TType limit_;
	TType replacement_;
};


struct LocalMinimumLabel
{
	// TODO
	template<typename TAccessor>
	CUGIP_DECL_HYBRID int
	operator()(TAccessor aAccessor) const
	{
		auto value = aAccessor[typename TAccessor::diff_t()];
		int res = 0;
		if (
			value <= aAccessor[typename TAccessor::diff_t(-1,-1)] &&
			value <= aAccessor[typename TAccessor::diff_t(-1,0)] &&
			value <= aAccessor[typename TAccessor::diff_t(-1,1)] &&
			value <= aAccessor[typename TAccessor::diff_t(1,-1)] &&
			value <= aAccessor[typename TAccessor::diff_t(1,0)] &&
			value <= aAccessor[typename TAccessor::diff_t(1,1)] &&
			value <= aAccessor[typename TAccessor::diff_t(0,1)] &&
			value <= aAccessor[typename TAccessor::diff_t(0,-1)])
		{
			res = get_linear_access_index(aAccessor.dimensions(), aAccessor.coords()) + 1;
			//printf("val = %d - %d\n", value, res);
			//return get_linear_access_index(aAccessor.dimensions(), aAccessor.coords()) + 1;
		} /*else {
			return 0;
		}*/
		/*printf("vals = %d, %d: %d %d %d %d %d %d %d %d [%d, %d] - [%d, %d]\n", value, res,
			aAccessor[typename TAccessor::diff_t(-1,-1)],
			aAccessor[typename TAccessor::diff_t(-1,0)],
			aAccessor[typename TAccessor::diff_t(-1,1)],
			aAccessor[typename TAccessor::diff_t(1,-1)],
			aAccessor[typename TAccessor::diff_t(1,0)],
			aAccessor[typename TAccessor::diff_t(1,1)],
			aAccessor[typename TAccessor::diff_t(0,1)],
			aAccessor[typename TAccessor::diff_t(0,-1)],
			aAccessor.dimensions()[0], aAccessor.dimensions()[1],
			aAccessor.coords()[0], aAccessor.coords()[1]
		);*/
		return res;
	}
};

struct HasSmallerNeighbor
{
	template<typename TAccessor>
	CUGIP_DECL_HYBRID int
	operator()(TAccessor aAccessor) const
	{
		auto value = aAccessor[typename TAccessor::diff_t()];
		if (
			value > aAccessor[typename TAccessor::diff_t(-1,-1)] ||
			value > aAccessor[typename TAccessor::diff_t(-1,0)] ||
			value > aAccessor[typename TAccessor::diff_t(-1,1)] ||
			value > aAccessor[typename TAccessor::diff_t(1,-1)] ||
			value > aAccessor[typename TAccessor::diff_t(1,0)] ||
			value > aAccessor[typename TAccessor::diff_t(1,1)] ||
			value > aAccessor[typename TAccessor::diff_t(0,1)] ||
			value > aAccessor[typename TAccessor::diff_t(0,-1)])
		{
			return 1;
		} else {
			return 0;
		}
	}
};

struct ZipValues
{
	template<typename... TTypes>
	CUGIP_DECL_HYBRID Tuple<TTypes...>
	operator()(TTypes... aArgs) const
	{
		return Tuple<TTypes...>(aArgs...);
	}
};

}//namespace cugip
