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

/*template<typename TStructuralTensor>
struct compute_diffusion_tensor_ftor
{
	CUGIP_DECL_HYBRID TStructuralTensor
	operator()(const TStructuralTensor &aTensor)
	{
		//TODO - 3d version
		TStructuralTensor tensor;
		float a = get<0>(aTensor);
		float b = get<1>(aTensor);
		float d = get<2>(aTensor);
		float rtD = sqrtf(sqr(a-d) + 4*sqr(b));
		float l1 = 0.5f * ((a+d) + rtD);
		float l2 = 0.5f * ((a+d) - rtD);

		vect2f_t v1(b, (l1-a));
		vect2f_t v2(b, (l2-a));
		
		v1 = normalize(v1);
		v2 = normalize(v2);

		float coh = sqr(l1-l2);
		float alpha = 0.01;
		float C = 1;
		l1 = alpha;
		if (coh < EPSILON) {
			l2 = alpha;
		} else {
			l2 = alpha + (1-alpha)*expf(-C/coh);
		}
		//if (abs(l2) / abs(l1) < 0.3) { l1 = 1.0f; l2 = 0.05; } else { l1 = l2 = 1.0f; }
	
		get<0>(tensor) = l1*v1[0]*v1[0] + l2*v2[0]*v2[0];
		get<1>(tensor) = l1*v1[0]*v1[1] + l2*v2[0]*v2[1];
		get<2>(tensor) = l1*v1[1]*v1[1] + l2*v2[1]*v2[1];
		return tensor;
	}
};*/

template<typename TStructuralTensor>
struct compute_diffusion_tensor_ftor
{
	CUGIP_DECL_HYBRID TStructuralTensor
	operator()(const TStructuralTensor &aTensor)
	{
		//TODO - 3d version
		TStructuralTensor tensor;
		float s11 = get<0>(aTensor);
		float s12 = get<1>(aTensor);
		float s22 = get<2>(aTensor);

		float s1_m_s2 = s11 - s22;

		float gamma = 0.02f;
		float C = 1.0f;
		float alfa = sqrtf(sqr(s1_m_s2) + 4*sqr(s12))+0.00001f;

		float c1 = gamma;
      		float c2 = gamma + (1-gamma)*expf(-C/(alfa));

		float c1_p_c2 = c1 + c2;
      		float c2_m_c1 = c2 - c1;
      
		float dd = (c2_m_c1 * s1_m_s2 )/(alfa);
      
      		get<0>(tensor) = 0.5f * ( c1_p_c2 + dd );
      		get<1>(tensor) = -(c2_m_c1)*s12/(alfa);
      		get<2>(tensor) = 0.5f * (c1_p_c2 - dd);

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
	convolution(aStructuralTensor, aBluredStructuralTensor, gaussian_kernel<float, size_traits_2d<9,9> >());
}

	
template<typename TTensorView>
void compute_diffusion_tensor(TTensorView aDiffusionTensor)
{
	for_each(aDiffusionTensor, cugip::compute_diffusion_tensor_ftor<typename TTensorView::value_type>());
}

template<typename TStructuralTensor, typename TGradient>
struct apply_diffusion_tensor_ftor
{
	CUGIP_DECL_HYBRID void
	operator()(const TStructuralTensor &aTensor, TGradient &aGradient)
	{
		//TODO - generic dimension
		TGradient tmp(0);
		get<0>(tmp) = get<0>(aTensor) * get<0>(aGradient) + get<1>(aTensor) * get<1>(aGradient);
		get<1>(tmp) = get<1>(aTensor) * get<0>(aGradient) + get<2>(aTensor) * get<1>(aGradient);
		
		aGradient = tmp;
	}
};

template<typename TTensorView, typename TGradientView>
void apply_diffusion_tensor(TTensorView aDiffusionTensor, TGradientView aGradient)
{
	for_each(aDiffusionTensor, aGradient, cugip::apply_diffusion_tensor_ftor<typename TTensorView::value_type, typename TGradientView::value_type>());
}

template<typename TGradientView, typename TOutputView>
void compute_diffusion_step(TGradientView aGradient, TOutputView aOuput)
{
	filter(aGradient, aOuput, cugip::divergence<typename TGradientView::value_type, typename TOutputView::value_type>());
}

template<typename TInView, typename TGradientView, typename TOutView, typename TTensorView>
void
coherence_enhancing_diffusion_step(
                TInView     aInput, 
                TOutView    aOuput, 
                TGradientView aGradient, 
                TTensorView aStructuralTensor, 
                TTensorView aDiffusionTensor
                )
{
	//TODO - check requirements
	
	compute_gradient(aInput, aGradient);

	compute_structural_tensor(aGradient, aStructuralTensor);

	blur_structural_tensor(aStructuralTensor, aDiffusionTensor);

	compute_diffusion_tensor(aDiffusionTensor);

	apply_diffusion_tensor(aDiffusionTensor, aGradient);

	compute_diffusion_step(aGradient, aOuput);
}

} //cugip
