#pragma once

#include <cugip/basic_filters/gradient.hpp>
#include <cugip/filter.hpp>

namespace cugip {

template<typename TInView, typename TGradientView>
void compute_gradient(TInView aInput, TGradientView aGradient)
{
	filter(aInput, aGradient, cugip::gradient_symmetric_difference<typename TInView::value_type, typename TGradientView::value_type>());
}

template<typename TGradientVector, typename TStructuralTensor>
struct compute_structural_tensor_ftor
{
	CUGIP_DECL_HYBRID TStructuralTensor
	operator()(const TGradientVector &aGradient)
	{
		TStructuralTensor tensor;
		get<0>(tensor) =  get<0>(aGradient) * get<0>(aGradient);
		get<1>(tensor) =  get<0>(aGradient) * get<1>(aGradient);
		get<2>(tensor) =  get<1>(aGradient) * get<1>(aGradient);
		return tensor;
	}
};

template<typename TStructuralTensor>
struct compute_diffusion_tensor_ftor
{
	CUGIP_DECL_HYBRID TStructuralTensor
	operator()(const TStructuralTensor &aTensor)
	{
		TStructuralTensor tensor;
		float a = get<0>(aTensor);
		float b = get<1>(aTensor);
		float d = get<2>(aTensor);
		float rtD = sqrt((a-d)*(a-d) + 4*b*b);
		float l1 = 0.5f * ((a+d) + rtD);
		float l2 = 0.5f * ((a+d) - rtD);

		vect2f_t v1(b, (l1-a));
		vect2f_t v2(b, (l2-a));
		
		v1 = normalize(v1);
		v2 = normalize(v2);

		//if (abs(l2) / abs(l1) < 0.3) { l1 = 1.0f; l2 = 0.05; } else { l1 = l2 = 1.0f; }
	
		get<0>(tensor) = l1*v1[0]*v1[0] + l2*v2[0]*v2[0];
		get<1>(tensor) = l1*v1[0]*v1[1] + l2*v2[0]*v2[1];
		get<2>(tensor) = l1*v1[1]*v1[1] + l2*v2[1]*v2[1];
		return tensor;
	}
};

template<typename TGradientView, typename TTensorView>
void compute_structural_tensor(TGradientView aGradient, TTensorView aStructuralTensor)
{
	transform(aGradient, aStructuralTensor, cugip::compute_structural_tensor_ftor<typename TGradientView::value_type, typename TTensorView::value_type>());
}

template<typename TInputTensorView, typename TOutputTensorView>
void blur_structural_tensor(TInputTensorView aStructuralTensor, TOutputTensorView aBluredStructuralTensor)
{
	convolution(aStructuralTensor, aBluredStructuralTensor, gaussian_kernel<float, size_traits_2d<7,7> >());
}

	
template<typename TTensorView>
void compute_diffusion_tensor(TTensorView aDiffusionTensor)
{
	for_each(aDiffusionTensor, cugip::compute_diffusion_tensor_ftor<typename TTensorView::value_type>());
}


template<typename TInView, typename TGradientView, typename TOutView, typename TTensorView>
void
coherence_enhancing_diffusion_step(
                TInView     aInput, 
                TOutView    aOuput, 
                TGradientView aGradient, 
                TTensorView aStructuralTensor, 
                TTensorView aDiffusionTensor,
                float       aStepSize
                )
{
	//TODO - check requirements
	
	compute_gradient(aInput, aGradient);

	compute_structural_tensor(aGradient, aStructuralTensor);

	blur_structural_tensor(aStructuralTensor, aDiffusionTensor);

	compute_diffusion_tensor(aDiffusionTensor);

	/*apply_diffusion_step(aInput, aOuput, aGradient, aDiffusionTensor, aStepSize);*/	
}

} //cugip
