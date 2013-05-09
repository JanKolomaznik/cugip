#pragma once

namespace cugip {

namespace watershed {
namespace detail {

} //namespace detail


template <typename TImageView, typename TMarkersView, typename TDistanceView>
void 
watershed_transformation(
		TImageView aImageView, 
		TMarkersView aMarkersView, 
		TMarkersView aMarkersView2, 
		TDistanceView aDistanceView, 
		TDistanceView aDistanceView2)
{
	detail::init_distance_buffer(aMarkersView, aDistanceView);

	device_flag updateFlag;
	do {
		detail::watershed_evolution(
				aImageView,
				aMarkersView,
				aMarkersView2,
				aDistanceView,
				aDistanceView2
				);
		using std::swap;
		swap(aMarkersView, aMarkersView2);
		swap(aDistanceView, aDistanceView2);
	} while (updateFlag);

}

} //namespace watershed


} //namespace cugip
