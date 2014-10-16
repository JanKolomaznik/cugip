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
watershed_transformation(TImageView aImageView, TIdImageView aIdImageView)
{

	step1(aImageView, aIdImageView);
	step2(aImageView, aIdImageView);
	step3(aImageView, aIdImageView);
	step4(aImageView, aIdImageView);
/*
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
	D_PRINT("Watershed transform done!");*/
}


}//namespace cugip
