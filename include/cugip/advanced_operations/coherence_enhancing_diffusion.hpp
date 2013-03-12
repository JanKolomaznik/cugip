#pragma once

namespace cugip {

template<typename TInView, typename TGradientView, typename TOutView, typename TTensorView>
void
coherence_enhancing_diffusion_step(
                TInView     aInput, 
                TOutView    aOuput, 
                TGradientView aGradient, 
                TTensorView aStructuralTensor, 
                TTensorView aBluredStructuralTensor, 
                TTensorView aDiffusionTensor
                float       aStepSize
                )
{
	//TODO - check requirements
	
	compute_gradient(aInput, aGradient);

	compute_structural_tensor(aGradient, aStructuralTensor);

	blur_structural_tensor(aStructuralTensor, aBluredStructuralTensor);

	compute_diffusion_tensor(aBluredStructuralTensor, aDiffusionTensor);

	apply_diffusion_step(aInput, aOuput, aGradient, aDiffusionTensor, aStepSize);	
}

} //cugip
