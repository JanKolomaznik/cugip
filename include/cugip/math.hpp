#pragma once

//#include <boost/array.hpp>
 #include <boost/type_traits.hpp>
 #include <cugip/traits.hpp>
 #include <cugip/utils.hpp>
 #include <cmath>

namespace cugip {


template<typename TCoordinateType, size_t tDim>
class simple_vector//: public boost::array<TCoordinateType, tDim>
{
public:
	typedef TCoordinateType coord_t;
	static const size_t dim = tDim;
    /*inline CUGIP_DECL_HYBRID simple_vector()
    {}*/

	inline CUGIP_DECL_HYBRID simple_vector(TCoordinateType const& v0 = 0, TCoordinateType const& v1 = 0, TCoordinateType const& v2 = 0)
	{
		if (tDim >= 1) mValues[0] = v0;
		if (tDim >= 2) mValues[1] = v1;
		//TODO
		//if (tDim >= 3) mValues[2] = v2;
	}

	template<typename TOtherCoordType>
	CUGIP_DECL_HYBRID simple_vector(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (size_t i = 0; i < tDim; ++i) {
			mValues[i] = aArg.mValues[i];
		}
	}

	template<typename TOtherCoordType>
	inline CUGIP_DECL_HYBRID simple_vector &
	operator=(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (size_t i = 0; i < tDim; ++i) {
			mValues[i] = aArg.mValues[i];
		}
		return *this;
	}

	template<typename TOtherCoordType>
	inline CUGIP_DECL_HYBRID simple_vector &
	operator+=(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (size_t i = 0; i < tDim; ++i) {
			mValues[i] += aArg.mValues[i];
		}
		return *this;
	}

	template<typename TOtherCoordType>
	inline CUGIP_DECL_HYBRID simple_vector &
	operator-=(const simple_vector<TOtherCoordType, tDim> &aArg)
	{
		for (size_t i = 0; i < tDim; ++i) {
			mValues[i] -= aArg.mValues[i];
		}
		return *this;
	}

	template <size_t tIdx>
	inline CUGIP_DECL_HYBRID TCoordinateType const& 
	get() const
	{
		//BOOST_STATIC_ASSERT(tIdx < DimensionCount);
		return mValues[tIdx];
	}

	template <size_t tIdx>
	inline CUGIP_DECL_HYBRID TCoordinateType & 
	get()
	{
		//BOOST_STATIC_ASSERT(tIdx < DimensionCount);
		return mValues[tIdx];
	}

	inline CUGIP_DECL_HYBRID const TCoordinateType & 
	operator[](size_t aIdx)const
	{
		return mValues[aIdx];
	}

	inline CUGIP_DECL_HYBRID TCoordinateType & 
	operator[](size_t aIdx)
	{
		return mValues[aIdx];
	}

	template <size_t tIdx>
	inline CUGIP_DECL_HYBRID void  
	set(TCoordinateType const& value)
	{
		//BOOST_STATIC_ASSERT(tIdx < tDim);
		mValues[tIdx] = value;
	}

	TCoordinateType mValues[tDim];
};

template<typename TCoordType1, typename TCoordType2, size_t tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim>
operator+(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim> res(aArg1);
	return res += aArg2;
}

template<typename TCoordType1, typename TCoordType2, size_t tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim>
operator*(const TCoordType1 &aFactor, const simple_vector<TCoordType2, tDim> &aArg)
{
	simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim> res;
	for (size_t i = 0; i < tDim; ++i) {
		res.mValues[i] = aFactor * aArg.mValues[i];
	}
	return res;
}

template<typename TCoordType1, typename TCoordType2, size_t tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim>
operator-(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim> res(aArg1);
	return res -= aArg2;
}

template<typename TCoordType1, typename TCoordType2, size_t tDim>
inline CUGIP_DECL_HYBRID bool
operator==(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	bool res = true;
	for (size_t i = 0; i < tDim && res; ++i) {
		res = aArg1.mValues[i] == aArg2.mValues[i];
	}
	return res;
}

template<typename TCoordType1, typename TCoordType2, size_t tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim>
max_coords(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim> res;
	for (size_t i = 0; i < tDim; ++i) {
		res.mValues[i] = aArg1.mValues[i] > aArg2.mValues[i] ? aArg1.mValues[i] : aArg2.mValues[i];
	}
	return res;
}

template<typename TCoordType1, typename TCoordType2, size_t tDim>
inline CUGIP_DECL_HYBRID simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim>
min_coords(const simple_vector<TCoordType1, tDim> &aArg1, const simple_vector<TCoordType2, tDim> &aArg2)
{
	simple_vector<typename boost::common_type<TCoordType1, TCoordType2>::type, tDim> res;
	for (size_t i = 0; i < tDim; ++i) {
		res.mValues[i] = aArg1.mValues[i] < aArg2.mValues[i] ? aArg1.mValues[i] : aArg2.mValues[i];
	}
	return res;
}

template<typename TType>
CUGIP_DECL_HOST inline std::ostream &
operator<<( std::ostream &stream, const simple_vector<TType,2> &v )
{
	return stream << "[ " << v. template get<0>() << ", " << v. template get<1>() << " ]";
}

template<typename TCoordType, size_t tDim>
inline CUGIP_DECL_HYBRID TCoordType
dot_product(const simple_vector<TCoordType, tDim> &aVector1, const simple_vector<TCoordType, tDim> &aVector2)
{
	//TODO - optimize
	TCoordType ret = 0;
	for (size_t i = 0; i < tDim; ++i) {
		ret += aVector1[i] * aVector2[i];
	}
	return ret;
}

template <typename TType>
inline CUGIP_DECL_HYBRID typename TType::coord_t
magnitude(const TType &aVector)
{
	return sqrtf(dot_product(aVector, aVector));
}

template <typename TType>
inline CUGIP_DECL_HYBRID TType
normalize(const TType &aVector)
{
	return (1.0f/magnitude(aVector)) * aVector;
}


/** \ingroup auxiliary_function
 * @{
 **/
/*template<size_t tIdx, typename TCoordinateType, int tDim>
CUGIP_DECL_HYBRID const TCoordinateType &
get(const simple_vector<TCoordinateType, tDim> &aArg)
{
	return aArg.get<tIdx>();
}


template<size_t tIdx, typename TCoordinateType, int tDim>
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

/** \ingroup traits
 * @{
 **/

//-----------------------------------------------------------------------------
template<size_t tIdx, typename TType, size_t tChannelCount>
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

template<size_t tIdx, typename TType, size_t tChannelCount>
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
template<size_t tDim>
struct dim_traits
{
	///Signed integral vector - used as a offset vector
	typedef simple_vector<int, tDim> diff_t;

	///Unsigned integral vector - used for defining sizes
	typedef simple_vector<size_t, tDim> extents_t;

	///Signed integral coordinate vector
	typedef simple_vector<int, tDim> coord_t;

	extents_t create_extents_t(size_t v0 = 0, size_t v1 = 0, size_t v2 = 0, size_t v3 = 0)
	{//TODO
		return extents_t(v0, v1/*, v2, v3*/);
	}
};
//-----------------------------------------------------------------------------


template<typename TCoordinateType, size_t tDim>
struct dimension<simple_vector<TCoordinateType, tDim> >: dimension_helper<tDim> {};

/** 
 * @}
 **/

#define EPSILON 0.000001f;

template<typename TType>
inline TType 
sqr(TType aValue) {
	return aValue * aValue;
}

}//namespace cugip
