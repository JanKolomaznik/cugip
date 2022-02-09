#pragma once

#include <cugip/detail/defines.hpp>

namespace cugip {

template<bool tFlag>
struct BoolValue
{
	static constexpr bool value = tFlag;
};

template<int tValue>
struct IntValue
{
	static constexpr int value = tValue;
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
	static constexpr int value = 1;
};

template<>
struct dimension_helper<2>
{
	typedef dimension_2d_tag type;
	static constexpr int value = 2;
};

template<>
struct dimension_helper<3>
{
	typedef dimension_3d_tag type;
	static constexpr int value = 3;
};

template<>
struct dimension_helper<4>
{
	typedef dimension_4d_tag type;
	static constexpr int value = 4;
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
	static constexpr int dimension = 1;
	static constexpr int size = tWidth;
	static constexpr int width = tWidth;

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
	static constexpr int dimension = 2;
	static constexpr int size = tWidth * tHeight;
	static constexpr int width = tWidth;
	static constexpr int height = tHeight;

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
	static constexpr int dimension = 3;

	static constexpr int size = tWidth * tHeight * tDepth;
	static constexpr int width = tWidth;
	static constexpr int height = tHeight;
	static constexpr int depth = tDepth;

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

template<typename TView>
struct is_array_view: public std::false_type {};

template<typename TView>
struct is_device_view: public std::false_type {};

template<typename TView>
struct is_host_view: public std::false_type {};

template<typename TView>
struct is_memory_based: public std::false_type {};

template<typename TView>
struct is_image_view: public std::false_type {};

template<typename TView>
struct is_interpolated_view: public std::false_type {};


/**
 * @}
 **/

}//namespace cugip
