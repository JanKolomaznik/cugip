#pragma once

 #include <type_traits>

 #include <cugip/traits.hpp>
 #include <cugip/utils.hpp>
 #include <cmath>

namespace cugip {

template<typename TType>
CUGIP_DECL_HYBRID constexpr TType
zero()
{
	return TType();
}

template<>
inline constexpr int zero<int>()
{
	return 0;
}

template<>
inline constexpr float zero<float>()
{
	return 0.0f;
}
//TODO handle other types properly


template<typename TCoordinateType, int tDim>
class simple_vector//: public boost::array<TCoordinateType, tDim>
{
public:
	typedef TCoordinateType coord_t;
	typedef simple_vector<TCoordinateType, tDim> this_t;
	static const int dim = tDim;

	CUGIP_DECL_HYBRID constexpr
	simple_vector()
		: mValues{ zero<TCoordinateType>() }
	{
		/*for (int i = 0; i < tDim; ++i) {
			mValues[i] = 0;
		}*/
	}

	CUGIP_DECL_HYBRID constexpr explicit
	simple_vector(TCoordinateType const& v0)
		: mValues{ v0 }
	{
		static_assert(tDim == 1, "Dimension must be 1!");
		//mValues[0] = v0;
		/*for (int i = 1; i < tDim; ++i) {
			mValues[i] = 0;
		}*/
	}

	CUGIP_DECL_HYBRID constexpr
	simple_vector(TCoordinateType const& v0, TCoordinateType const& v1)
		: mValues{ v0, v1 }
	{
		static_assert(tDim == 2, "Dimension must be 2!");
		//mValues[0] = v0;
		//mValues[1] = v1;
		/*for (int i = 2; i < tDim; ++i) {
			mValues[i] = 0;
		}*/
	}

	CUGIP_DECL_HYBRID constexpr
	simple_vector(TCoordinateType const& v0, TCoordinateType const& v1, TCoordinateType const& v2)
		: mValues{ v0, v1, v2 }
	{
		static_assert(tDim == 3, "Dimension must be 3!");
		//mValues[0] = v0;
		//mValues[1] = v1;
		//mValues[2] = v2;
		/*for (int i = 3; i < tDim; ++i) {
			mValues[i] = 0;
		}*/
	}

	template<typename TOtherCoordType>
	CUGIP_DECL_HYBRID /*constexpr*/
	simple_vector(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (int i = 0; i < tDim; ++i) {
			mValues[i] = aArg.mValues[i];
		}
	}

	template<typename TOtherCoordType>
	inline CUGIP_DECL_HYBRID simple_vector &
	operator=(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (int i = 0; i < tDim; ++i) {
			mValues[i] = aArg.mValues[i];
		}
		return *this;
	}

	template<typename TOtherCoordType>
	inline CUGIP_DECL_HYBRID simple_vector &
	operator+=(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (int i = 0; i < tDim; ++i) {
			mValues[i] += aArg.mValues[i];
		}
		return *this;
	}

	template<typename TOtherCoordType>
	inline CUGIP_DECL_HYBRID simple_vector &
	operator-=(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (int i = 0; i < tDim; ++i) {
			mValues[i] -= aArg.mValues[i];
		}
		return *this;
	}

	template <int tIdx>
	inline CUGIP_DECL_HYBRID TCoordinateType const&
	get() const
	{
		//BOOST_STATIC_ASSERT(tIdx < DimensionCount);
		return mValues[tIdx];
	}

	template <int tIdx>
	inline CUGIP_DECL_HYBRID TCoordinateType &
	get()
	{
		//BOOST_STATIC_ASSERT(tIdx < DimensionCount);
		return mValues[tIdx];
	}

	inline CUGIP_DECL_HYBRID const TCoordinateType &
	operator[](int aIdx)const
	{
		return mValues[aIdx];
	}

	inline CUGIP_DECL_HYBRID TCoordinateType &
	operator[](int aIdx)
	{
		return mValues[aIdx];
	}

	template <int tIdx>
	inline CUGIP_DECL_HYBRID void
	set(TCoordinateType const& value)
	{
		//BOOST_STATIC_ASSERT(tIdx < tDim);
		mValues[tIdx] = value;
	}


	static CUGIP_DECL_HYBRID this_t
	fill(TCoordinateType aValue)
	{
		this_t val;
		for (int i = 0; i < tDim; ++i) {
			val[i] = aValue;
		}
		return val;
	}

	TCoordinateType mValues[tDim];
};

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim>
operator+(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim> res(aArg1);
	return res += aArg2;
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim>
operator*(const TCoordType1 &aFactor, const simple_vector<TCoordType2, tDim> &aArg)
{
	simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim> res;
	for (int i = 0; i < tDim; ++i) {
		res.mValues[i] = aFactor * aArg.mValues[i];
	}
	return res;
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim>
operator-(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim> res(aArg1);
	return res -= aArg2;
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID bool
operator==(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	for (int i = 0; i < tDim; ++i) {
		if (aArg1.mValues[i] != aArg2.mValues[i]) {
			return false;
		}
	}
	return true;
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID bool
operator!=(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	return !(aArg1 == aArg2);
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID bool
less( const simple_vector<TCoordType1, tDim> &aA, const simple_vector<TCoordType2, tDim> &aB)
{
	for (int i = 0; i < tDim; ++i) {
		if (aA.mValues[i] >= aB.mValues[i]) {
			return false;
		}
	}
	return true;
}


template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID bool
operator<( const simple_vector<TCoordType1, tDim> &aA, const simple_vector<TCoordType2, tDim> &aB)
{
	return less(aA, aB);
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID bool
operator<=( const simple_vector<TCoordType1, tDim> &aA, const simple_vector<TCoordType2, tDim> &aB)
{
	for (int i = 0; i < tDim; ++i) {
		if (aA.mValues[i] > aB.mValues[i]) {
			return false;
		}
	}
	return true;
}


template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID bool
operator>=( const simple_vector<TCoordType1, tDim> &aA, const simple_vector<TCoordType2, tDim> &aB)
{
	return aB <= aA;
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim>
max_coords(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim> res;
	for (int i = 0; i < tDim; ++i) {
		res.mValues[i] = aArg1.mValues[i] > aArg2.mValues[i] ? aArg1.mValues[i] : aArg2.mValues[i];
	}
	return res;
}

template<typename TCoordType1, typename TCoordType2, int tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim>
min_coords(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim> res;
	for (int i = 0; i < tDim; ++i) {
		res.mValues[i] = aArg1.mValues[i] < aArg2.mValues[i] ? aArg1.mValues[i] : aArg2.mValues[i];
	}
	return res;
}

template<typename TType, int tDim>
CUGIP_DECL_HOST inline std::ostream &
operator<<(std::ostream &stream, const simple_vector<TType, tDim> &v)
{
	stream << "[ ";
	for (int i = 0; i < tDim - 1; ++i) {
		stream << v[i] << ", ";
	}
	return stream << v[tDim - 1] << " ]";
}

template<typename TCoordType, int tDim>
inline CUGIP_DECL_HYBRID TCoordType
dot(const simple_vector<TCoordType, tDim> &aVector1, const simple_vector<TCoordType, tDim> &aVector2)
{
	//TODO - optimize
	TCoordType ret = 0;
	for (int i = 0; i < tDim; ++i) {
		ret += aVector1[i] * aVector2[i];
	}
	return ret;
}

template<typename TCoordType, int tDim>
inline CUGIP_DECL_HYBRID simple_vector<TCoordType, tDim>
product(simple_vector<TCoordType, tDim> aVector1, const simple_vector<TCoordType, tDim> &aVector2)
{
	//TODO - optimize
	for (int i = 0; i < tDim; ++i) {
		aVector1[i] *= aVector2[i];
	}
	return aVector1;
}


template<typename TType>
CUGIP_DECL_HYBRID typename std::enable_if<std::is_fundamental<TType>::value, TType>::type
dot(const TType &aValue1, const TType &aValue2)
{
	return aValue1 * aValue2;
}

template<typename TCoordType>
inline CUGIP_DECL_HYBRID simple_vector<TCoordType, 3>
cross(const simple_vector<TCoordType, 3> &aVector1, const simple_vector<TCoordType, 3> &aVector2)
{
	simple_vector<TCoordType, 3> result;

	result[0] = aVector1[1] * aVector2[2] - aVector1[2] * aVector2[1];
	result[1] = aVector1[2] * aVector2[0] - aVector1[0] * aVector2[2];
	result[2] = aVector1[0] * aVector2[1] - aVector1[1] * aVector2[0];
	return result;
}

template <typename TType>
inline CUGIP_DECL_HYBRID typename TType::coord_t
squared_magnitude(const TType &aVector)
{
	return dot(aVector, aVector);
}

template <typename TType>
inline CUGIP_DECL_HYBRID typename TType::coord_t
magnitude(const TType &aVector)
{
	return sqrtf(squared_magnitude(aVector));
}

template <typename TType>
inline CUGIP_DECL_HYBRID TType
normalize(const TType &aVector)
{
	return (1.0f/magnitude(aVector)) * aVector;
}

template<typename TCoordType, int tDim>
inline CUGIP_DECL_HYBRID TCoordType
multiply(const simple_vector<TCoordType, tDim> &aVector)
{
	TCoordType ret = 1;
	for (int i = 0; i < tDim; ++i) {
		ret *= aVector[i];
	}
	return ret;
}

/** \ingroup auxiliary_function
 * @{
 **/
/*template<int tIdx, typename TCoordinateType, int tDim>
CUGIP_DECL_HYBRID const TCoordinateType &
get(const simple_vector<TCoordinateType, tDim> &aArg)
{
	return aArg.get<tIdx>();
}


template<int tIdx, typename TCoordinateType, int tDim>
CUGIP_DECL_HYBRID TCoordinateType &
get(simple_vector<TCoordinateType, tDim> &aArg)
{
	return aArg.get<tIdx>();
}*/


/**
 * @}
 **/

typedef simple_vector<float, 2> intervalf_t;
typedef simple_vector<double, 2> intervald_t;

typedef simple_vector<float, 2> vect2f_t;
typedef simple_vector<float, 3> vect3f_t;

typedef simple_vector<int, 2> vect2i_t;
typedef simple_vector<int, 3> vect3i_t;

typedef simple_vector<int, 2> size2_t;
typedef simple_vector<int, 3> size3_t;

typedef simple_vector<int, 2> Int2;
typedef simple_vector<int, 3> Int3;
/** \ingroup traits
 * @{
 **/

//-----------------------------------------------------------------------------
template<int tIdx, typename TType, int tChannelCount>
struct get_policy<tIdx, const simple_vector<TType, tChannelCount> >
{
	typedef const TType & return_type;
	typedef const simple_vector<TType, tChannelCount> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		return aArg.template get<tIdx>();
	}
};

template<int tIdx, typename TType, int tChannelCount>
struct get_policy<tIdx, simple_vector<TType, tChannelCount> >
{
	typedef TType & return_type;
	typedef simple_vector<TType, tChannelCount> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		return aArg.template get<tIdx>();
	}
};
//-----------------------------------------------------------------------------
template<int tDim>
struct dim_traits
{
	///Signed integral vector - used as a offset vector
	typedef simple_vector<int, tDim> diff_t;

	///Unsigned integral vector - used for defining sizes
	typedef simple_vector<int, tDim> extents_t;

	///Signed integral coordinate vector
	typedef simple_vector<int, tDim> coord_t;

	extents_t create_extents_t(int v0 = 0, int v1 = 0, int v2 = 0, int v3 = 0)
	{//TODO
		CUGIP_ASSERT(false);
		return extents_t(v0, v1/*, v2, v3*/);
	}
};

template<>
struct dim_traits<1>
{
	typedef int diff_t;

	///Unsigned integral vector - used for defining sizes
	typedef int extents_t;

	///Signed integral coordinate vector
	typedef int coord_t;

	extents_t create_extents_t(int v0 = 0, int v1 = 0, int v2 = 0, int v3 = 0)
	{//TODO
		return extents_t(v0);
	}
};
//-----------------------------------------------------------------------------


template<typename TCoordinateType, int tDim>
struct dimension<simple_vector<TCoordinateType, tDim> >: dimension_helper<tDim> {};

/**
 * @}
 **/

#define EPSILON 0.000001f

template<typename TType>
inline CUGIP_DECL_HYBRID TType
sqr(TType aValue) {
	return aValue * aValue;
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
abs(TType aValue) {
	if (aValue < 0) {
		return -1 * aValue;
	}
	return aValue;
}

template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID simple_vector<TType, tDimension>
abs(simple_vector<TType, tDimension> aValue) {
	simple_vector<TType, tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = abs(aValue[i]);
	}
	return result;
}


template<typename TType>
inline CUGIP_DECL_HYBRID TType
max(TType aValue1, TType aValue2) {
	return aValue1 < aValue2 ? aValue2 : aValue1;
}

template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID TType
max(simple_vector<TType, tDimension> aValue) {
	auto maxValue = aValue[0];
	for (int i = 1; i < tDimension; ++i) {
		maxValue = max(aValue[i], maxValue);
	}
	return maxValue;
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
min(TType aValue1, TType aValue2) {
	return aValue1 < aValue2 ? aValue1 : aValue2;
}

template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID TType
min(simple_vector<TType, tDimension> aValue) {
	auto minValue = aValue[0];
	for (int i = 1; i < tDimension; ++i) {
		minValue = min(aValue[i], minValue);
	}
	return minValue;
}

template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID simple_vector<TType, tDimension>
min_per_element(simple_vector<TType, tDimension> aVector1, const simple_vector<TType, tDimension> &aVector2) {
	for (int i = 0; i < tDimension; ++i) {
		aVector1[i] = min(aVector1[i], aVector2[i]);
	}
	return aVector1;
}


template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID simple_vector<TType, tDimension>
max_per_element(simple_vector<TType, tDimension> aVector1, const simple_vector<TType, tDimension> &aVector2) {
	for (int i = 0; i < tDimension; ++i) {
		aVector1[i] = max(aVector1[i], aVector2[i]);
	}
	return aVector1;
}

template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID TType
sum(const simple_vector<TType, tDimension> &aVector)
{
	TType result = aVector[0];
	for (int i = 1; i < tDimension; ++i) {
		result += aVector[i];
	}
	return result;
}

template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID simple_vector<TType, tDimension>
div(const simple_vector<TType, tDimension> &aVector1, TType aValue)
{
	simple_vector<TType, tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = aVector1[i] / aValue;
	}
	return result;
}

template<typename TType, int tDimension>
inline CUGIP_DECL_HYBRID simple_vector<TType, tDimension>
div(const simple_vector<TType, tDimension> &aVector1, const simple_vector<TType, tDimension> &aVector2)
{
	simple_vector<TType, tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = aVector1[i] / aVector2[i];
	}
	return result;
}


/// \return All vector elements product
template<typename TType, int tDimension>
CUGIP_DECL_HYBRID TType
product(const simple_vector<TType, tDimension> &aVector)
{
	TType result = 1;
	for (int i = 0; i < tDimension; ++i) {
		result *= aVector[i];
	}
	return result;
}

template<typename TType>
CUGIP_DECL_HYBRID typename std::enable_if<std::is_fundamental<TType>::value, TType>::type
product(const TType &aValue)
{
	return aValue;
}

template<typename TCoordType, int tDim>
inline CUGIP_DECL_HYBRID void
sort(simple_vector<TCoordType, tDim> &aVector)
{
	//TODO - check performance
	for (int i = 0; i < tDim; ++i) {
		for (int j = 1; i < tDim - i; ++i) {
			if (aVector[j] < aVector[j-1]) {
				swap(aVector[j], aVector[j-1]);
			}
		}
	}
}

template<typename TType>
inline CUGIP_DECL_HYBRID simple_vector<TType, 3>
find_orthonormal(const simple_vector<TType, 3> &aVector)
{
	simple_vector<TType, 3> result;
	for (int j = 0; j < 3; ++j) {
		// find some orthogonal vector to the first eigen vector
		if (aVector[j] != 0.0f) {
			// swap non-zero coordinate with following one and clear the third -> perpendicular vector
			auto norm = 1.0f / sqrt(sqr(aVector[j]) + sqr(aVector[(j + 1) % 3]));
			result[j] = aVector[(j + 1) % 3] * norm;
			result[(j + 1) % 3] = -aVector[j] * norm;
			result[(j + 2) % 3] = 0.0f;
			return result;
		}
	}
	return result;
}



}//namespace cugip
