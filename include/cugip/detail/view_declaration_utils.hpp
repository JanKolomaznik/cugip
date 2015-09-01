#pragma once

#include <cugip/utils.hpp>

#define REMOVE_PARENTHESES(...) __VA_ARGS__

#define CUGIP_DECLARE_DEVICE_VIEW_TRAITS(CLASS, DIMENSION,...)\
	template<__VA_ARGS__>\
	struct is_device_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct dimension<REMOVE_PARENTHESES CLASS>: dimension_helper<DIMENSION> {};

#define CUGIP_DECLARE_HOST_VIEW_TRAITS(CLASS, DIMENSION, ...)\
	template<__VA_ARGS__>\
	struct is_host_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct dimension<REMOVE_PARENTHESES CLASS>: dimension_helper<DIMENSION> {};

#define CUGIP_DECLARE_HYBRID_VIEW_TRAITS(CLASS, DIMENSION, ...)\
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

/**
 * @}
 **/
 
CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID int
elementCount(const TView &aView)
{
	return product(aView.dimensions());
}

}//namespace cugip
