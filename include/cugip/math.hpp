#pragma once

//#include <boost/array.hpp>
 #include <boost/type_traits.hpp>

namespace cugip {


template<typename TCoordinateType, size_t tDim>
class simple_vector//: public boost::array<TCoordinateType, tDim>
{
public:
	typedef TCoordinateType coord_t;
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

template<typename TType, size_t tIdx>
CUGIP_DECL_HYBRID const typename TType::coord_t &
get(const TType &aArg)
{
	return aArg.get<tIdx>();
}


template<size_t tDim>
struct dim_traits
{
	typedef simple_vector<int, tDim> diff_t;

	typedef simple_vector<size_t, tDim> extents_t;

	typedef simple_vector<int, tDim> coord_t;

	extents_t create_extents_t(size_t v0 = 0, size_t v1 = 0, size_t v2 = 0, size_t v3 = 0)
	{//TODO
		return extents_t(v0, v1/*, v2, v3*/);
	}
};

typedef simple_vector<float, 2> intervalf_t;
typedef simple_vector<double, 2> intervald_t;


}//namespace cugip
