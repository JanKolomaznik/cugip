#pragma once

#include <cugip/multiresolution_pyramid.hpp>
#include <cugip/host_image.hpp>
#include <cugip/host_image_view.hpp>

namespace cugip {

struct ObjectnessMeasureConfig
{

};

struct PostProcessAlgorithm
{

};

template<typename TInputView, typename TPyramidView, typename TTmpView>
void hessian_eigen_value_pyramid(TInputView aInput, TPyramidView aOutput, TTmpView aTmpView, ObjectnessMeasureConfig aConfig)
{

	//template<typename TInputView, typename TPyramidView, typename TTmpView, typename TAlgorithm>
	compute_multiresolution_pyramid(aInput, aOutput, aTmpView, PostProcessAlgorithm());
}

template<typename TInputView, typename TOutputView>
void objectness_measure(TInputView aInput, TOutputView aOutput, ObjectnessMeasureConfig aConfig)
{
	using Image = host_image<typename TInputView::value_type, dimension<TInputView>::value>;
	multiresolution_pyramid<Image> pyramid;
	host_image<typename TInputView::value_type, dimension<TInputView>::value> tmpImage;
	hessian_eigen_value_pyramid(aInput, view(pyramid), view(tmpImage), aConfig);
}

} //namespace cugip
