#pragma once

//#include <boost/array.hpp>

namespace cugip {


template<typename TCoordinateType, size_t tDim>
class simple_vector//: public boost::array<TCoordinateType, tDim>
{
public:
    inline simple_vector()
    {}

    inline simple_vector(TCoordinateType const& v0, TCoordinateType const& v1 = 0, TCoordinateType const& v2 = 0)
    {
        if (tDim >= 1) mValues[0] = v0;
        if (tDim >= 2) mValues[1] = v1;
	//TODO
        //if (tDim >= 3) mValues[2] = v2;
    }

    inline simple_vector(const simple_vector &aArg)
    {
	for (size_t i = 0; i < tDim; ++i) {
		mValues[i] = aArg.mValues[i];
	}
    }

    inline simple_vector &
    operator=(const simple_vector &aArg)
    {
	for (size_t i = 0; i < tDim; ++i) {
		mValues[i] = aArg.mValues[i];
	}
	return *this;
    }

    template <size_t tIdx>
    inline TCoordinateType const& get() const
    {
        //BOOST_STATIC_ASSERT(tIdx < DimensionCount);
        return mValues[tIdx];
    }

    template <size_t tIdx>
    inline void set(TCoordinateType const& value)
    {
        //BOOST_STATIC_ASSERT(tIdx < tDim);
        mValues[tIdx] = value;
    }

private:

    TCoordinateType mValues[tDim];
};


template<size_t tDim>
struct dim_traits
{
	typedef simple_vector<int, tDim> diff_t;

	typedef simple_vector<size_t, tDim> extents_t;

	extents_t create_extents_t(size_t v0 = 0, size_t v1 = 0, size_t v2 = 0, size_t v3 = 0)
	{//TODO
		return extents_t(v0, v1/*, v2, v3*/);
	}
};


}//namespace cugip
