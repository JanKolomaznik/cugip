#pragma once

#include <cugip/basic_filters/gradient.hpp>
#include <cugip/math/symmetric_tensor.hpp>
#include <cugip/math/eigen.hpp>
#include <cugip/basic_filters/convolution.hpp>
#include <cugip/transform.hpp>
#include <cugip/for_each.hpp>

namespace cugip {

template<typename TInView, typename TGradientView>
void compute_gradient(TInView aInput, TGradientView aGradient)
{
	cugip::transform_locator(aInput, aGradient, sobel_gradient<3>());
}

template<typename TGradientVector, typename TStructuralTensor>
struct compute_structural_tensor_ftor
{
	static constexpr int cDimension = static_vector_traits<TGradientVector>::dimension;

	CUGIP_DECL_HYBRID TStructuralTensor
	operator()(const TGradientVector &aGradient)
	{
		TStructuralTensor tensor;
		for (int i = 0; i < cDimension; ++i) {
			for (int j = i; j < cDimension; ++j) {
				tensor.get(i, j) = aGradient[i] * aGradient[j];
			}
		}
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

struct compute_diffusion_tensor_ftor
{
	static constexpr float cKappaEpsilon = 0.00000001f;

	template<typename TStructuralTensor>
	CUGIP_DECL_HYBRID TStructuralTensor
	operator()(const TStructuralTensor &aTensor)
	{
		constexpr int cDimension = TStructuralTensor::cDimension;

		auto eigValues = eigen_values(aTensor);
		auto eigVectors = eigen_vectors(aTensor, eigValues);

		float kappa = 0.0f;
		for (int i = 0; i < cDimension - 1; ++i) {
			for (int j = i + 1; j < cDimension; ++j) {
				kappa += sqr(eigValues[i] - eigValues[j]);
			}
		}

		simple_vector<float, cDimension> newEigValues;

		for (int i = 0; i < cDimension; ++i) {
			newEigValues[i] = alpha;
		}
		if (kappa < cKappaEpsilon) {
			newEigValues[cDimension - 1] += (1 - alpha) * exp(-contrast / kappa);
		}

		return matrix_from_eigen_vectors_and_values(eigVectors, newEigValues);
	}

	float alpha;
	float contrast;
};

template<typename TGradientView, typename TTensorView>
void compute_structural_tensor(TGradientView aGradient, TTensorView aStructuralTensor)
{
	transform(aGradient, aStructuralTensor, cugip::compute_structural_tensor_ftor<typename TGradientView::value_type, typename TTensorView::value_type>());
}

template<typename TInputTensorView, typename TOutputTensorView, typename TTmpTensorView>
void blur_structural_tensor(TInputTensorView aStructuralTensor, TOutputTensorView aBluredStructuralTensor, TTmpTensorView aTmpTensorView)
{
	//cugip::separable_convolution(aStructuralTensor, aBluredStructuralTensor, aTmpTensorView, gaussian_kernel<5>());
	cugip::separable_convolution(aStructuralTensor, aBluredStructuralTensor, aTmpTensorView, gaussian_kernel<2>());
	//convolution(aStructuralTensor, aBluredStructuralTensor, gaussian_kernel<float, intraits_2d<9,9> >());
}


template<typename TTensorView>
void compute_diffusion_tensor(TTensorView aDiffusionTensor, float aAlpha, float aContrast)
{
	for_each(aDiffusionTensor, cugip::compute_diffusion_tensor_ftor{ aAlpha, aContrast });
}

struct apply_diffusion_tensor_ftor
{
	template<typename TStructuralTensor, typename TGradient>
	CUGIP_DECL_HYBRID TGradient
	operator()(const TStructuralTensor &aTensor, TGradient &aGradient)
	{
		return product(aTensor, aGradient);
	}
};

template<typename TTensorView, typename TGradientView, typename TOutputView>
void apply_diffusion_tensor(TTensorView aDiffusionTensor, TGradientView aGradient, TOutputView aOutput, float aTimeStep)
{
	transform2(aDiffusionTensor, aGradient, aGradient, cugip::apply_diffusion_tensor_ftor());
	transform_locator_assign(aGradient, aOutput, sobel_weighted_divergence<dimension<TTensorView>::value>(aTimeStep), transform_update_add());
}

/*template<typename TGradientView, typename TOutputView>
void compute_diffusion_step(TGradientView aGradient, TOutputView aOuput)
{
	filter(aGradient, aOuput, cugip::divergence<typename TGradientView::value_type, typename TOutputView::value_type>());
}*/

template<typename TInView, typename TGradientView, typename TOutView, typename TTensorView>
void
coherence_enhancing_diffusion_step(
                TInView     aInput,
                TOutView    aOutput,
                TGradientView aGradient,
                TTensorView aStructuralTensor,
                TTensorView aDiffusionTensor,
                TTensorView aTmpTensor,
		float aTimeStep,
		float aAlpha,
		float aContrast
                )
{
	//TODO - check requirements

	compute_gradient(aInput, aGradient);

	compute_structural_tensor(aGradient, aStructuralTensor);

	blur_structural_tensor(aStructuralTensor, aDiffusionTensor, aTmpTensor);

	compute_diffusion_tensor(aDiffusionTensor, aAlpha, aContrast);

	apply_diffusion_tensor(aDiffusionTensor, aGradient, aOutput, aTimeStep);

	//compute_diffusion_step(aGradient, aOuput);
}

template<int tDimension>
class coherence_enhancing_diffusion
{
public:
	typedef symmetric_tensor<float, tDimension> StructuralTensor;
	typedef simple_vector<float, tDimension> Gradient;
	typedef simple_vector<int, tDimension> Size;

	coherence_enhancing_diffusion(Size aSize, float aTimeStep, float aAlpha, float aContrast)
		: mTimeStep(aTimeStep)
		, mAlpha(aAlpha)
		, mContrast(aContrast)
		, mGradient(aSize)
		, mStructuralTensor(aSize)
		, mDiffusionTensor(aSize)
		, mTmpTensor(aSize)
	{}

	template<typename TInView, typename TOutView>
	void iteration(TInView aInput, TOutView aOutput)
	{
		coherence_enhancing_diffusion_step(
			aInput,
			aOutput,
			view(mGradient),
			view(mStructuralTensor),
			view(mDiffusionTensor),
			view(mTmpTensor),
			mTimeStep,
			mAlpha,
			mContrast);

	}

	float mTimeStep;
	float mAlpha;
	float mContrast;

	device_image<Gradient, tDimension> mGradient;
	device_image<StructuralTensor, tDimension> mStructuralTensor;
	device_image<StructuralTensor, tDimension> mDiffusionTensor;
	device_image<StructuralTensor, tDimension> mTmpTensor;
};

} //cugip
