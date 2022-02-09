#pragma once

#include <type_traits>
#include <limits>
#include <cmath>


namespace cugip {



#define EPSILON 0.000001f

template<typename TType>
inline constexpr CUGIP_DECL_HYBRID TType
sqr(TType aValue) {
	return aValue * aValue;
}

template<typename TType>
inline constexpr CUGIP_DECL_HYBRID TType
pow_n(TType aValue, int aPower) {
	/*switch (aPower) {
	case 0:
		return { 1 };
	case 1:
		return aValue;
	*/
	TType result = {1};
	for (int i = 1; i <= aPower; ++i) {
		result *= aValue;
	}
	return result;
}

inline CUGIP_DECL_HYBRID int
round(float aVal)
{
#ifdef __CUDA_ARCH__
	return roundf(aVal);
#else
	return std::round(aVal);
#endif
}


inline CUGIP_DECL_HYBRID int
floor(float aVal)
{
#ifdef __CUDA_ARCH__
	return floorf(aVal);
#else
	return std::floor(aVal);
#endif
}

inline CUGIP_DECL_HYBRID int
ceil(float aVal)
{
#ifdef __CUDA_ARCH__
	return ceilf(aVal);
#else
	return std::ceil(aVal);
#endif
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
abs(TType aValue) {
	if (aValue < 0) {
		return -1 * aValue;
	}
	return aValue;
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
max(TType aValue1, TType aValue2) {
	return aValue1 < aValue2 ? aValue2 : aValue1;
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
min(TType aValue1, TType aValue2) {
	return aValue1 < aValue2 ? aValue1 : aValue2;
}

template<typename TType>
inline CUGIP_DECL_HYBRID int
signum(TType aValue) {
    	int t = aValue < 0 ? -1 : 0;
    	return aValue > 0 ? 1 : t;
}

template<typename TType>
inline CUGIP_DECL_HYBRID auto
multiply(const TType &aValue)
{
	static_assert(std::is_arithmetic_v<TType>);
	return aValue;
}

template<typename TType1, typename TType2>
inline CUGIP_DECL_HYBRID auto
div_up(const TType1 &aValue1, const TType2 &aValue2)
{
	return (aValue1 + aValue2 - 1) / aValue2;
}
template<typename TTo, typename TFrom>
CUGIP_DECL_HYBRID TTo
safe_numeric_cast(TFrom aVal) {
	if constexpr(std::numeric_limits<TFrom>::lowest() < std::numeric_limits<TTo>::lowest()) {
		if (aVal < std::numeric_limits<TTo>::lowest()) {
			aVal = std::numeric_limits<TTo>::lowest();
		}
	}
	if constexpr(std::numeric_limits<TFrom>::max() > std::numeric_limits<TTo>::max()) {
		if (aVal > std::numeric_limits<TTo>::max()) {
			aVal = std::numeric_limits<TTo>::max();
		}
	}
	// TODO rounding;

	return TTo(aVal);
}


}//namespace cugip
