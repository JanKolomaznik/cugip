#pragma once

#include <cugip/basic_filters/gradient.hpp>
#include <cugip/filter.hpp>

namespace cugip {

template<typename TInView, typename TGradientView>
void compute_gradient(TInView aInput, TGradientView aGradient)
{
	filter(aInput, aGradient, cugip::gradient_symmentric_difference<typename TInView::value_type, typename TGradientView::value_type>());
}

template<typename TGradientVector, typename TStructuralTensor>
struct compute_structural_tensor_ftor
{
	CUGIP_DECL_HYBRID TStructuralTensor
	operator()(const TGradientVector &aGradient)
	{
		TStructuralTensor tensor;
		get<0>(tensor) =  get<0>(aGradient) * get<0>(aGradient);
		get<1>(tensor) =  get<1>(aGradient) * get<1>(aGradient);
		get<2>(tensor) =  get<0>(aGradient) * get<1>(aGradient);
		return tensor;
	}
};

template<typename TGradientView, typename TTensorView>
void compute_structural_tensor(TGradientView aGradient, TTensorView aStructuralTensor)
{
	transform(aGradient, aStructuralTensor, cugip::compute_structural_tensor_ftor<typename TGradientView::value_type, typename TTensorView::value_type>());
}

template<typename TInView, typename TGradientView, typename TOutView, typename TTensorView>
void
coherence_enhancing_diffusion_step(
                TInView     aInput, 
                TOutView    aOuput, 
                TGradientView aGradient, 
                TTensorView aStructuralTensor, 
                TTensorView aBluredStructuralTensor, 
                TTensorView aDiffusionTensor,
                float       aStepSize
                )
{
	//TODO - check requirements
	
	compute_gradient(aInput, aGradient);

	compute_structural_tensor(aGradient, aStructuralTensor);

	/*blur_structural_tensor(aStructuralTensor, aBluredStructuralTensor);

	compute_diffusion_tensor(aBluredStructuralTensor, aDiffusionTensor);

	apply_diffusion_step(aInput, aOuput, aGradient, aDiffusionTensor, aStepSize);*/	
}

} //cugip
