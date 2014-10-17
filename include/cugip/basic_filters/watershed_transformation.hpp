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
	typedef typename TIdImageView::locator LabelLocator;
	typedef typename TDistanceView::locator Locator;
	typedef typename TDistanceView::value_type Distance;

	typename TImageView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TImageView::extents_t extents = aImageView.dimensions();

	if (coord < extents) {
		Value value = aImageView[coord];
		Locator distanceLocator = aDistanceView1.template locator<cugip::border_handling_repeat_t>(coord);
		Distance currentDistance = distanceLocator.get();
		LabelLocator labelLocator = aIdImageView.template locator<cugip::border_handling_repeat_t>(coord);
		Label label = labelLocator.get();

		Distance currentMinimum = max(0, currentDistance - value);
		typename Locator::diff_t offset;
		bool found = false;
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				Distance distance = distanceLocator[typename Locator::diff_t(i, j)];
				if (distance < currentMinimum) {
					currentMinimum = distance;
					offset = typename Locator::diff_t(i, j);
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


	init_watershed_buffers(aIdImageView, aDistanceView1);
	while (updatedFlag.check_host()) {
		watershed_evolution(
				aImageView,
				aIdImageView,
				aTmpIdImageView,
				aDistanceView1,
				aDistanceView2,
				updatedFlag.view());
	}
}



}//namespace cugip
