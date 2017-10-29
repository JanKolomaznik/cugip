#pragma once

#include <cugip/utils.hpp>

namespace cugip {

template<int tValue>
struct IntValue
{
	static const int value = tValue;
};

struct dimension_1d_tag { };
struct dimension_2d_tag { };
struct dimension_3d_tag { };
struct dimension_4d_tag { };

template<int tDim>
struct dimension_helper;

template<>
struct dimension_helper<1>
{
	typedef dimension_1d_tag type;
	static const int value = 1;
};

template<>
struct dimension_helper<2>
{
	typedef dimension_2d_tag type;
	static const int value = 2;
};

template<>
struct dimension_helper<3>
{
	typedef dimension_3d_tag type;
	static const int value = 3;
};

template<>
struct dimension_helper<4>
{
	typedef dimension_4d_tag type;
	static const int value = 4;
};


/** \addtogroup traits
 *  Traits
 * @{
 **/
template<typename TType>
struct dimension;


template <int tWidth>
struct intraits_1d
{
	static const int dimension = 1;
	static const int size = tWidth;
	static const int width = tWidth;

	template <typename TCoords>
	CUGIP_DECL_HYBRID static int
	get_index(const TCoords &aCoords)
	{
		return aCoords[0];
	}

	template <typename TCoords>
	CUGIP_DECL_HYBRID static TCoords
	get_extents()
	{
		return TCoords(tWidth);
	}
};

template <int tWidth, int tHeight>
struct intraits_2d
{
	static const int dimension = 2;
	static const int size = tWidth * tHeight;
	static const int width = tWidth;
	static const int height = tHeight;

	template <typename TCoords>
	CUGIP_DECL_HYBRID static int
	get_index(const TCoords &aCoords)
	{
		return width * aCoords[1] + aCoords[0];
	}

	template <typename TCoords>
	CUGIP_DECL_HYBRID static TCoords
	get_extents()
	{
		return TCoords(tWidth, tHeight);
	}
};

template <int tWidth, int tHeight, int tDepth>
struct intraits_3d
{
	static const int dimension = 3;

	static const int size = tWidth * tHeight * tDepth;
	static const int width = tWidth;
	static const int height = tHeight;
	static const int depth = tDepth;

	template <typename TCoords>
	CUGIP_DECL_HYBRID static int
	get_index(const TCoords &aCoords)
	{
		return (width * height) * aCoords[2] + width * aCoords[1] + aCoords[0];
	}

	template <typename TCoords>
	CUGIP_DECL_HYBRID static TCoords
	get_extents()
	{
		return TCoords(tWidth, tHeight, tDepth);
	}
};

template <int tWidth>
struct dimension<intraits_1d<tWidth> >: dimension_helper<1>
{};

template <int tWidth, int tHeight>
struct dimension<intraits_2d<tWidth, tHeight> >: dimension_helper<2>
{};

template <int tWidth, int tHeight, int tDepth>
struct dimension<intraits_3d<tWidth, tHeight, tDepth> >: dimension_helper<3>
{};

/**
 * @}
 **/

}//namespace cugip
