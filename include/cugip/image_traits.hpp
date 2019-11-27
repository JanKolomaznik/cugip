#pragma once

#include <cugip/utils.hpp>
#include <cugip/traits.hpp>

#include <cugip/host_image.hpp>

#if defined(__CUDACC__)
#include <cugip/image.hpp>
#include <cugip/unified_image.hpp>
#endif //defined(__CUDACC__)

#include <cugip/detail/view_declaration_utils.hpp>

namespace cugip {


template<typename TView>
struct image_view_traits {
	//static_assert(is_image_view<TView>::value, "TView must be follow image view concept.");

	static constexpr int dimension = cugip::dimension<TView>::value;
	using value_type = typename TView::value_type;

	using host_image_t = host_image<value_type, dimension>;


#if defined(__CUDACC__)
	using device_image_t = cugip::device_image<value_type, dimension>;
	using unified_image_t = cugip::unified_image<value_type, dimension>;
#else
	using device_image_t = void;
	using unified_image_t = void;
#endif //defined(__CUDACC__)


	using image_t = typename std::conditional<
		is_device_view<TView>::value,
		typename std::conditional<
			is_host_view<TView>::value,
			unified_image_t,
			device_image_t>::type,
		host_image_t>::type;

};

/**
 * @}
 **/

}//namespace cugip

