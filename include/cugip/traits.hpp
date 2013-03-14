#pragma once

namespace cugip {

struct dimension_1d_tag { };
struct dimension_2d_tag { };
struct dimension_3d_tag { };
struct dimension_4d_tag { };

template<size_t tDim>
struct dimension_helper;

template<>
struct dimension_helper<1>
{
	typedef dimension_1d_tag type;
	static const size_t value = 1;
};

template<>
struct dimension_helper<2>
{
	typedef dimension_2d_tag type;
	static const size_t value = 2;
};

template<>
struct dimension_helper<3>
{
	typedef dimension_3d_tag type;
	static const size_t value = 3;
};

template<>
struct dimension_helper<4>
{
	typedef dimension_4d_tag type;
	static const size_t value = 4;
};


/** \defgroup traits
 * @{
 **/
template<typename TType>
struct dimension;


template <size_t tWidth>
struct size_traits_1d
{
	static const size_t dimension = 1;
	static const size_t size = tWidth;
	static const size_t width = tWidth;
};

template <size_t tWidth, size_t tHeight>
struct size_traits_2d
{
	static const size_t dimension = 2;
	static const size_t size = tWidth * tHeight;
	static const size_t width = tWidth;
	static const size_t height = tHeight;
};

template <size_t tWidth, size_t tHeight, size_t tDepth>
struct size_traits_3d
{
	static const size_t dimension = 3;

	static const size_t size = tWidth * tHeight * tDepth;
	static const size_t width = tWidth;
	static const size_t height = tHeight;
	static const size_t depth = tDepth;
};

template <size_t tWidth>
struct dimension<size_traits_1d<tWidth> >: dimension_helper<1>
{};

template <size_t tWidth, size_t tHeight>
struct dimension<size_traits_2d<tWidth, tHeight> >: dimension_helper<2>
{};

template <size_t tWidth, size_t tHeight, size_t tDepth>
struct dimension<size_traits_3d<tWidth, tHeight, tDepth> >: dimension_helper<3>
{};
/** 
 * @}
 **/

}//namespace cugip
