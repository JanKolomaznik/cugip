#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>

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

		size_t iteration = 0;
		size_t max_iteration = 1000;

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
	dim_traits<2>::extents_t extents;
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
struct gradient_difference
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

}//namespace cugip
