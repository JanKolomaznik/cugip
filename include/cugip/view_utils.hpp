#pragma once
#include <cugip/traits.hpp>
#include <cugip/region.hpp>

#include <boost/type_index.hpp>
#include <sstream>


namespace cugip {

CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID int64_t
elementCount(const TView &aView)
{
	static_assert(is_array_view<TView>::value || is_image_view<TView>::value, "Only arrays and images supported.");
	if constexpr (is_array_view<TView>::value) {
		return aView.size();
	}

	if constexpr (is_image_view<TView>::value) {
		return product(coord_cast<int64_t>(aView.dimensions()));
	}
	return 0;
}

CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID bool
isEmpty(const TView &aView)
{
	return 0 == elementCount(aView);
}

CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID
region<dimension<TView>::value> valid_region(const TView &view) {
	if constexpr (1 == dimension<TView>::value) {
		region<1>{0, view.size()};
	} else {
		return region<dimension<TView>::value>{
			typename TView::coord_t(),
			view.dimensions() };
	}
}

CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID
region<dimension<TView>::value> active_region(const TView &view) {
	if constexpr (1 == dimension<TView>::value) {
		return region<1>{0, view.size()};
	} else {
		return region<dimension<TView>::value>{
			typename TView::coord_t(),
			view.dimensions() };
	}
}


template<typename TView>
std::string
info(const TView &aView)
{
	// static_assert(is_array_view<TView>::value || is_image_view<TView>::value, "Only arrays and images supported.");
	std::ostringstream s;

	s << "{ \"type\": \"" << boost::typeindex::type_id<TView>().pretty_name() << "\"";

	if constexpr (is_array_view<TView>::value) {
		s << ", \"dimensions\": " << aView.size();
	}

	if constexpr (is_image_view<TView>::value) {
		s << ", \"dimensions\": " << aView.dimensions();
	}
	s << " }";
	return s.str();
}


} // namespace cugip
