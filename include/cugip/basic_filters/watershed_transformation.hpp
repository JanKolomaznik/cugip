#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/access_utils.hpp>
#include <cugip/basic_filters/connected_component_labeling.hpp>


namespace cugip {
/*
namespace detail {

template<typename TInputType, typename TLUT>
struct scan_neighborhood_for_steepest_descent_ftor
{
	scan_neighborhood_for_steepest_descent_ftor(TLUT aLUT, device_flag_view aLutUpdatedFlag)
		: mLUT(aLUT), mLutUpdatedFlag(aLutUpdatedFlag)
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID TInputType
	minValidLabel(TLocator aLocator, TInputType aCurrent, const dimension_2d_tag &)
	{
		TInputType minimum = aCurrent;
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				TInputType value = aLocator[typename TLocator::diff_t(i, j)];
				minimum = (value < minimum && value > 0) ? value : minimum;
			}
		}
		return minimum;
	}

	template<typename TLocator>
	CUGIP_DECL_DEVICE void // TODO hybrid
	operator()(TLocator aLocator)
	{
		TInputType current = aLocator.get();
		if (0 < current) {
			TInputType minLabel = minValidLabel(aLocator, current, typename dimension<TLocator>::type());;
			if (minLabel < mLUT[current-1]) {
				//printf("%d - %d - %d; ", current, mLUT[current-1], minLabel);
				mLUT[current-1] = minLabel;
				mLutUpdatedFlag.set_device();
			}
		}
	}

	TLUT mLUT;
	device_flag_view mLutUpdatedFlag;
};


template <typename TImageView, typename TLUTBufferView>
void
scan_image_for_steepest_descent_ftor(TImageView aImageView, TLUTBufferView aLUT, device_flag_view aLutUpdatedFlag)
{
	for_each_locator(aImageView, scan_neighborhood_for_connections_ftor<typename TImageView::value_type, TLUTBufferView>(aLUT, aLutUpdatedFlag));
}
//-----------------------------------------------------------------------------


}//namespace detail


template <typename TImageView, typename TIdImageView, typename TLUTBufferView>
void
watershed_transformation(TImageView aImageView, TIdImageView, TLUTBufferView aLUT)
{
	device_flag lutUpdatedFlag;
	lutUpdatedFlag.reset_host();

	D_PRINT("Watershed transform initialization ...");
	detail::init_lut(aImageView, aLUT);
	detail::scan_image(aImageView, aLUT, lutUpdatedFlag.view());

	int i = 0;
	while (lutUpdatedFlag.check_host()) {
		D_PRINT("    Running WShed iteration ..." << ++i);
		lutUpdatedFlag.reset_host();

		detail::update_lut(aImageView, aLUT);
		detail::update_labels(aImageView, aLUT);
		detail::scan_image(aImageView, aLUT, lutUpdatedFlag.view());
	}
	D_PRINT("Watershed transform done!");
}
*/
//*****************************************************************************************

//template <typename TImageView, typename TIdImageView,>
struct step1_ftor
{
	template<typename TImageLocator, typename TIdLocator>
	CUGIP_DECL_DEVICE void
	operator()(TImageLocator aImageLocator, TIdLocator aIdLocator)
	{
		typename TImageLocator::value_type currentMin = aImageLocator.get();
		bool lowerExists = false;
		typename TImageLocator::diff_t offset;
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				typename TImageLocator::value_type value = aImageLocator[typename TImageLocator::diff_t(i, j)];
				if (value < currentMin) {
					lowerExists = true;
					currentMin = value;
					offset = typename TImageLocator::diff_t(i, j);
				}
			}
		}
		if (lowerExists) {
			aIdLocator.get() = -aIdLocator[offset];
		} else {
			aIdLocator.get() = 0; //Plateau
		}
	}
};


template <typename TImageView, typename TIdImageView>
void
step1(TImageView aImageView, TIdImageView aIdImageView)
{
	for_each_locator(aImageView, aIdImageView, step1_ftor());
}

template <typename TImageView, typename TIdImageView>
void
step2(TImageView aImageView, TIdImageView aIdImageView)
{
	//for_each_locator(aImageView, aIdImageView, step1_ftor());
}

template <typename TImageView, typename TIdImageView>
void
step3(TImageView aImageView, TIdImageView aIdImageView)
{
	//for_each_locator(aImageView, aIdImageView, step1_ftor());
}

template <typename TImageView, typename TIdImageView>
void
step4(TImageView aImageView, TIdImageView aIdImageView)
{
	//for_each_locator(aImageView, aIdImageView, step1_ftor());
}


template <typename TImageView, typename TIdImageView>
void
watershed_transformation(TImageView aImageView, TIdImageView aIdImageView, TIdImageView aTmpIdImageView)
{
	step1(aImageView, aIdImageView);
	step2(aImageView, aIdImageView, aTmpIdImageView);
	step3(aImageView, aIdImageView);
	step4(aImageView, aIdImageView);
}
//**************************************************
template<typename TLabel, typename TDistance>
struct init_wsheds_ftor
{
	init_wsheds_ftor(TDistance aInfinity)
		: infinity(aInfinity)
	{}

	CUGIP_DECL_HYBRID void
	operator()(const TLabel &aLabel, TDistance &aDistance)
	{
		aDistance = aLabel == 0 ? infinity : 0;
	}

	TDistance infinity;
};

template <typename TIdImageView, typename TDistanceView>
void
init_watershed_buffers(TIdImageView aIdImageView, TDistanceView aDistanceView)
{
	for_each(
		aIdImageView,
		aDistanceView,
		init_wsheds_ftor<typename TIdImageView::value_type, typename TDistanceView::value_type>(65000)); //TODO
}
//**************************************************
template <typename TImageView, typename TIdImageView, typename TDistanceView>
CUGIP_GLOBAL void
watershed_evolution_kernel(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TIdImageView aTmpIdImageView,
		TDistanceView aDistanceView1,
		TDistanceView aDistanceView2,
		device_flag_view aUpdateFlag)
{
	typedef typename TImageView::value_type Value;
	typedef typename TIdImageView::value_type Label;
	typedef image_locator<TIdImageView, cugip::border_handling_repeat_t> LabelLocator;
	typedef image_locator<TDistanceView, cugip::border_handling_repeat_t> DistanceLocator;
	typedef typename TDistanceView::value_type Distance;

	typename TImageView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TImageView::extents_t extents = aImageView.dimensions();

	if (coord < extents) {
		Value value = aImageView[coord];
		DistanceLocator distanceLocator = aDistanceView1.template locator<cugip::border_handling_repeat_t>(coord);
		Distance currentDistance = distanceLocator.get();
		LabelLocator labelLocator = aIdImageView.template locator<cugip::border_handling_repeat_t>(coord);
		Label label = labelLocator.get();

		Distance currentMinimum = max(0, currentDistance - value);
		typename DistanceLocator::diff_t offset;
		bool found = false;
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				Distance distance = distanceLocator[typename DistanceLocator::diff_t(i, j)];
				if (distance < currentMinimum) {
					currentMinimum = distance;
					offset = typename DistanceLocator::diff_t(i, j);
					found = true;
				}
			}
		}
		if (found) {
			aUpdateFlag.set_device();
			currentDistance = currentMinimum + value;
			label = labelLocator[offset];
		}
		aDistanceView2[coord] = currentDistance;
		aTmpIdImageView[coord] = label;
	}

}

template <typename TImageView, typename TIdImageView, typename TDistanceView>
void
watershed_evolution(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TIdImageView aTmpIdImageView,
		TDistanceView aDistanceView1,
		TDistanceView aDistanceView2,
		device_flag_view aUpdateFlag)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aImageView.dimensions().template get<0>() / blockSize.x + 1), aImageView.dimensions().template get<1>() / blockSize.y + 1, 1);

	watershed_evolution_kernel<<<gridSize, blockSize>>>(
				aImageView,
				aIdImageView,
				aTmpIdImageView,
				aDistanceView1,
				aDistanceView2,
				aUpdateFlag);
}


template <typename TImageView, typename TIdImageView, typename TDistanceView>
void
watershed_transformation2(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TIdImageView aTmpIdImageView,
		TDistanceView aDistanceView1,
		TDistanceView aDistanceView2
		)
{
	device_flag updatedFlag;
	updatedFlag.reset_host();

	D_PRINT("    Init WSHED computation ...");
	init_watershed_buffers(aIdImageView, aDistanceView1);

	int i = 0;
	do {
		D_PRINT("    Running WSHED iteration ..." << ++i);
		updatedFlag.reset_host();
		watershed_evolution(
				aImageView,
				aIdImageView,
				aTmpIdImageView,
				aDistanceView1,
				aDistanceView2,
				updatedFlag.view());
		std::swap(aTmpIdImageView, aIdImageView);
		std::swap(aDistanceView1, aDistanceView2);
		//TODO handle copying after last iteration
	} while (updatedFlag.check_host());
}
//==================================================================================

//**************************************************
template<typename TUnionFind, typename TExtents>
struct init_watershed_labels_ftor
{
	init_watershed_labels_ftor(TUnionFind aUnionFind, TExtents aExtents)
		: unionFind(aUnionFind)
		, extents(aExtents)
	{}

	template<typename TLabel, typename TCoordinates>
	CUGIP_DECL_HYBRID TLabel
	operator()(const TLabel &aLabel, const TCoordinates &aCoords)
	{
		TLabel label = 1 + get_linear_access_index(extents, aCoords);
		unionFind.set(label, label);
		return label;
	}

	TUnionFind unionFind;
	TExtents extents;
};

template <typename TIdImageView, typename TUnionFind>
void
init_watershed_labels(TIdImageView aIdImageView, TUnionFind aUnionFind)
{
	for_each_position(
		aIdImageView,
		init_watershed_labels_ftor<TUnionFind, typename TIdImageView::extents_t>(aUnionFind, aIdImageView.dimensions())
		);
}

//**************************************************
template<typename TUnionFind, typename TExtents>
struct init_watershed_labels_from_plateaus_ftor
{
	init_watershed_labels_from_plateaus_ftor(TUnionFind aUnionFind, TExtents aExtents)
		: unionFind(aUnionFind)
		, extents(aExtents)
	{}

	template<typename TLabel, typename TCoordinates>
	CUGIP_DECL_HYBRID TLabel
	operator()(const TLabel &aLabel, const TCoordinates &aCoords)
	{
		if (aLabel != 0) {
			return aLabel;
		}
		TLabel label = 1 + get_linear_access_index(extents, aCoords);
		unionFind.set(label, label);
		return label;
	}

	TUnionFind unionFind;
	TExtents extents;
};

template <typename TIdImageView, typename TUnionFind>
void
init_watershed_from_plateaus_labels(TIdImageView aIdImageView, TUnionFind aUnionFind)
{
	for_each_position(
		aIdImageView,
		init_watershed_labels_from_plateaus_ftor<TUnionFind, typename TIdImageView::extents_t>(aUnionFind, aIdImageView.dimensions())
		);
}



//-----------------------------------------------------------------------------

template<typename TUnionFind>
struct scan_image_for_wshed_path_ftor
{
	scan_image_for_wshed_path_ftor(TUnionFind aUnionFind, device_flag_view aUpdatedFlag)
		: mUnionFind(aUnionFind)
		, mUpdatedFlag(aUpdatedFlag)
	{}

	template<typename TImageLocator, typename TLabelImageLocator>
	CUGIP_DECL_DEVICE void // TODO hybrid
	operator()(TImageLocator aImageLocator, TLabelImageLocator aLabelLocator)
	{
		typedef typename TImageLocator::value_type Value;
		typedef typename TLabelImageLocator::value_type Label;
		Value current = aImageLocator.get();
		Value minimum = current;
		typename TImageLocator::diff_t offset;
		bool found = false;
		bool hasBigger = false;
		//bool isPlateau = true;
		Label currentLabel = aLabelLocator.get();
		bool print = (aImageLocator.coords()[0] == 133) && (aImageLocator.coords()[1] == 32);
	//	Label plateauLabel = aLabelLocator[typename TLabelImageLocator::diff_t(-1, -1)];
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				Value value = aImageLocator[typename TImageLocator::diff_t(i, j)];
				if (value > current) {
					hasBigger = true;
				}
				/*if (value != current) {
					isPlateau = false;*/
					if (value <= minimum && i != 0 && j != 0/* || hasBigger && value == minimum*/) {
						minimum = value;
						found = true;
						offset = typename TImageLocator::diff_t(i, j);
					}
				if (print) {
					printf("%d, %d, label = %d, val = %d; h = %d; o = %d %d \n", i, j, int(aLabelLocator[typename TImageLocator::diff_t(i, j)]), int(value), int(hasBigger), offset[0], offset[1]);
				}
			/*	} else {
					plateauLabel = min(plateauLabel, aLabelLocator[typename TImageLocator::diff_t(i, j)]);
				}*/
			}
		}

		/*if (isPlateau) {
			Label minLabel = min(plateauLabel, currentLabel);
			Label maxLabel = max(plateauLabel, currentLabel);
			if (minLabel < mUnionFind.get(currentLabel)) {
				mUnionFind.set(currentLabel, minLabel);
				mUpdatedFlag.set_device();
			}
		} else {*/
			if (found && (minimum < current || hasBigger)) {
				Label lowerLabel = aLabelLocator[offset];
				Label minLabel = min(lowerLabel, currentLabel);
				Label maxLabel = max(lowerLabel, currentLabel);
				if (minLabel < mUnionFind.get(maxLabel)) {
					mUnionFind.set(maxLabel, minLabel);
					mUpdatedFlag.set_device();
				}
			}
		//}
	}

	TUnionFind mUnionFind;
	device_flag_view mUpdatedFlag;
};


template <typename TImageView, typename TIdImageView, typename TUnionFind>
void
scan_image_for_wshed_path(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TUnionFind aUnionFind,
		device_flag_view aUpdatedFlag)
{
	for_each_locator(
		aImageView,
		aIdImageView,
		scan_image_for_wshed_path_ftor<TUnionFind>(aUnionFind, aUpdatedFlag));
}


template <typename TImageView, typename TIdImageView, typename TUnionFind>
void
watershed_transformation1(
		TImageView aImageView,
		TIdImageView aIdImageView,
		TUnionFind aUnionFind
		)
{
	device_flag updatedFlag;
	updatedFlag.reset_host();

	D_PRINT("    Init WSHED computation ...");

	filter(aImageView, aIdImageView, local_minima_detection_ftor<typename TIdImageView::value_type>());
	connected_component_labeling(aIdImageView, aUnionFind);

	init_watershed_from_plateaus_labels(aIdImageView, aUnionFind);

	//init_watershed_labels(aIdImageView, aUnionFind);
	scan_image_for_wshed_path(
		aImageView,
		aIdImageView,
		aUnionFind,
		updatedFlag.view());


	int i = 0;
	while (updatedFlag.check_host()) {
		D_PRINT("    Running WSHED iteration ..." << ++i);
		updatedFlag.reset_host();

		detail::update_lut(aIdImageView, aUnionFind);
		detail::update_labels(aIdImageView, aUnionFind);

		scan_image_for_wshed_path(
			aImageView,
			aIdImageView,
			aUnionFind,
			updatedFlag.view());
	}
}



}//namespace cugip
