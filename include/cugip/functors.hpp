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
	mandelbrot_ftor(dim_traits<2>::extents_t aExtents = dim_traits<2>::extents_t()): extents(aExtents)
	{}

	CUGIP_DECL_HYBRID element_rgb8_t
	operator()(const element_rgb8_t &aArg, dim_traits<2>::coord_t aCoordinates)const
	{
		float x0 = (float(aCoordinates.get<0>()) / extents.get<0>()) * 3.5f - 2.5f;
		float y0 = (float(aCoordinates.get<1>()) / extents.get<1>()) * 2.0f - 1.0f;
		float x = 0.0f;
		float y = 0.0f;

		size_t iteration = 0;
		size_t max_iteration = 10000;

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
		element_gray8_t res;
		res.data[0] = tmp / 4;
		return res;
	}
	dim_traits<2>::extents_t extents;
};

}//namespace cugip
