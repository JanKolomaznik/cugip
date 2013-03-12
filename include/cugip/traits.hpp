#pragma once

namespace cugip {

struct dimension_1d_tag { };
struct dimension_2d_tag { };
struct dimension_3d_tag { };
struct dimension_4d_tag { };

template<size_t tDim>
struct dimension_helper;

template<>
struct dimension_helper<1>;
{
	typedef dimension_1d_tag type;
	static const size_t value = 1;
};

template<>
struct dimension_helper<2>;
{
	typedef dimension_2d_tag type;
	static const size_t value = 2;
};

template<>
struct dimension_helper<3>;
{
	typedef dimension_3d_tag type;
	static const size_t value = 3;
};

template<>
struct dimension_helper<4>;
{
	typedef dimension_4d_tag type;
	static const size_t value = 4;
};


/** \defgroup traits
 * @{
 **/
template<typename TType>
struct dimension;

/** 
 * @}
 **/

}//namespace cugip
