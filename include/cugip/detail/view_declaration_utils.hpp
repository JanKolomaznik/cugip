#pragma once

#include <cugip/detail/defines.hpp>
#include <cugip/math.hpp>
#include <cugip/image_locator.hpp>
#include <cugip/region.hpp>


#define REMOVE_PARENTHESES(...) __VA_ARGS__

#define CUGIP_DECLARE_VIEW_TRAITS(CLASS, DIMENSION, IS_DEVICE, IS_HOST, ...)\
	template<__VA_ARGS__>\
	struct is_image_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct is_device_view<REMOVE_PARENTHESES CLASS> : public std::integral_constant<bool, IS_DEVICE> {};\
	template<__VA_ARGS__>\
	struct is_host_view<REMOVE_PARENTHESES CLASS> : public std::integral_constant<bool, IS_HOST> {};\
	template<__VA_ARGS__>\
	struct dimension<REMOVE_PARENTHESES CLASS>: dimension_helper<DIMENSION> {};

#define CUGIP_DECLARE_DEVICE_VIEW_TRAITS(CLASS, DIMENSION,...)\
	template<__VA_ARGS__>\
	struct is_image_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct is_device_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct dimension<REMOVE_PARENTHESES CLASS>: dimension_helper<DIMENSION> {};

#define CUGIP_DECLARE_HOST_VIEW_TRAITS(CLASS, DIMENSION, ...)\
	template<__VA_ARGS__>\
	struct is_image_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct is_host_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct dimension<REMOVE_PARENTHESES CLASS>: dimension_helper<DIMENSION> {};

#define CUGIP_DECLARE_HYBRID_VIEW_TRAITS(CLASS, DIMENSION, ...)\
	template<__VA_ARGS__>\
	struct is_image_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct is_device_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct is_host_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct dimension<REMOVE_PARENTHESES CLASS>: dimension_helper<DIMENSION> {};

namespace cugip {

/** \ingroup  traits
 * @{
 **/
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
CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID
region<dimension<TView>::value> valid_region(const TView &view) {
	return region<dimension<TView>::value>{
		typename TView::coord_t(),
		view.dimensions() };
}

CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID int
elementCount(const TView &aView)
{
	return product(aView.dimensions());
}

CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID bool
isEmpty(const TView &aView)
{
	return 0 == product(aView.dimensions());
}

template<int tDimension>
class device_image_view_base
{
public:
	typedef typename dim_traits<tDimension>::extents_t extents_t;

	device_image_view_base(extents_t dimensions)
		: mDimensions(dimensions)
	{}

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mDimensions; }

	extents_t mDimensions;
};

template<int tDimension>
class host_image_view_base
{
public:
	typedef typename dim_traits<tDimension>::extents_t extents_t;

	host_image_view_base(extents_t dimensions)
		: mDimensions(dimensions)
	{}

	extents_t
	dimensions() const
	{ return mDimensions; }

	extents_t mDimensions;
};

template<int tDimension>
class hybrid_image_view_base
{
public:
	typedef typename dim_traits<tDimension>::extents_t extents_t;

	hybrid_image_view_base(extents_t dimensions)
		: mDimensions(dimensions)
	{}

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mDimensions; }

	extents_t mDimensions;
};


template<int tDimension, typename TDerived>
class device_image_view_crtp
{
public:
	typedef typename dim_traits<tDimension>::extents_t extents_t;
	typedef typename dim_traits<tDimension>::coord_t coord_t;
	typedef typename dim_traits<tDimension>::diff_t diff_t;

	device_image_view_crtp(extents_t dimensions)
		: mDimensions(dimensions)
	{}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<TDerived, TBorderHandling>
	locator(coord_t aCoordinates) const
	{
		return image_locator<TDerived, TBorderHandling>(*const_cast<TDerived *>(static_cast<const TDerived *>(this)), aCoordinates);
	}

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mDimensions; }

	extents_t mDimensions;
};

template<int tDim>
struct dimension<device_image_view_base<tDim>>: dimension_helper<tDim> {};


template<int tDim>
struct dimension<hybrid_image_view_base<tDim>>: dimension_helper<tDim> {};

template<int tDim>
struct dimension<host_image_view_base<tDim>>: dimension_helper<tDim> {};

#define CUGIP_VIEW_TYPEDEFS_VALUE(ElementType, aDimension)\
	static constexpr int cDimension = aDimension;\
	typedef typename dim_traits<aDimension>::extents_t extents_t;\
	typedef typename dim_traits<aDimension>::coord_t coord_t;\
	typedef typename dim_traits<aDimension>::diff_t diff_t;\
	typedef ElementType value_type;\
	typedef const ElementType const_value_type;\
	typedef value_type accessed_type;

#define CUGIP_VIEW_TYPEDEFS_REFERENCE(ElementType, aDimension)\
	static constexpr int cDimension = aDimension;\
	typedef typename dim_traits<aDimension>::extents_t extents_t;\
	typedef typename dim_traits<aDimension>::coord_t coord_t;\
	typedef typename dim_traits<aDimension>::diff_t diff_t;\
	typedef ElementType value_type;\
	typedef const ElementType const_value_type;\
	typedef value_type &accessed_type;


template<typename TView>
void dump_view_to_file(TView view, std::string filename) {
	static_assert(is_host_view<TView>::value, "Dump to file works only for host views");

	std::ofstream out;
	out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	out.open(filename, std::ofstream::out | std::ofstream::binary);

	for (int64_t i = 0; i < elementCount(view); ++i) {
		auto element = linear_access(view, i);
		out.write(reinterpret_cast<const char *>(&element), sizeof(element));
	}
}

}//namespace cugip
