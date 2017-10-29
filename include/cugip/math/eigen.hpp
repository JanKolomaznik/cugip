#pragma once

#include <cugip/math.hpp>
#include <cugip/math/symmetric_tensor.hpp>
#include <cugip/math/matrix.hpp>

namespace cugip {

/** \addtogroup math
 * @{
 **/

// TODO handle constants properly
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

CUGIP_DECL_HYBRID simple_vector<float, 3>
eigen_values(const symmetric_tensor<float, 3> &aTensor)
{
	// Determine coefficients of characteristic poynomial. We write
	//       | a   d   f  |
	//  A =  | d*  b   e  |
	//       | f*  e*  c  |
	float de = get<0, 1>(aTensor) * get<1, 2>(aTensor);			 // d * e
	float dd = sqr(get<0,1>(aTensor));					 // d^2
	float ee = sqr(get<1,2>(aTensor));					 // e^2
	float ff = sqr(get<0,2>(aTensor));					 // f^2
	float m  = matrix_trace(aTensor);
	float c1 =
		get<0,0>(aTensor) * get<1,1>(aTensor) +
		get<0,0>(aTensor) * get<2,2>(aTensor) +
		get<1,1>(aTensor) * get<2,2>(aTensor) -
		(dd + ee + ff);       // a*b + a*c + b*c - d^2 - e^2 - f^2
	float c0 =
		get<2,2>(aTensor) * dd +
		get<0,0>(aTensor) * ee +
		get<1,1>(aTensor) * ff -
		get<0,0>(aTensor) * get<1,1>(aTensor)*get<2,2>(aTensor) -
		2.0 * get<0,2>(aTensor) * de; // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

	float p = sqr(m) - 3.0 * c1;
	float q = m * (p - (3.0/2.0) * c1) - (27.0/2.0) * c0;
	float sqrt_p = sqrt(fabs(p));

	float phi = 27.0 * (0.25 * sqr(c1) * (p - c1) + c0*(q + 27.0/4.0 * c0));
	phi = (1.0 / 3.0) * atan2(sqrt(fabs(phi)), q);

	float c = sqrt_p * cos(phi);
	float s = (1.0/M_SQRT3) * sqrt_p * sin(phi);

	simple_vector<float, 3> vals;
	vals[1]  = (1.0/3.0) * (m - c);
	vals[2]  = vals[1] + s;
	vals[0]  = vals[1] + c;
	vals[1] -= s;
	return vals;
}

template<int tCol1, int tCol2>
CUGIP_DECL_HYBRID simple_vector<float, 3>
eigen_vector(float aEigenValue, const symmetric_tensor<float, 3> &aMatrix, const float aThreshold)
{
	auto v1 = aMatrix.column(tCol1);
	v1[tCol1] -= aEigenValue;
	auto v2 = aMatrix.column(tCol2);
	v2[tCol2] -= aEigenValue;
	auto n1 = squared_magnitude(v1);
	auto n2 = squared_magnitude(v2);
	if (n1 <= aThreshold) {
		return simple_vector<float, 3>(1.0f, 0.0f, 0.0f);
	} else if (n2 <= aThreshold) {
		return simple_vector<float, 3>(0.0f, 1.0f, 0.0f);
	}
	auto result = cross(v1, v2);
	auto norm = squared_magnitude(result);
	auto threshold2 = sqr(64.0f * EPSILON) * n1 * n2;
	if (norm < threshold2) {
		// If angle between A[0] and A[1] is too small, don't use
		// cross product, but calculate v ~ (1, -A0/A1, 0)
		auto t = sqr(get<0, 1>(aMatrix));
		auto f = -get<0, 0>(aMatrix) / get<0, 1>(aMatrix);
		if (sqr(get<1, 1>(aMatrix) > t)) {
			t = sqr(get<1, 1>(aMatrix));
			f = -get<0, 1>(aMatrix) / get<1, 1>(aMatrix);
		}
		if (sqr(get<1, 2>(aMatrix) > t)) {
			f = -get<0, 2>(aMatrix) / get<1, 2>(aMatrix);
		}
		norm = 1.0f / sqrt(1.0f + sqr(f));
		return simple_vector<float, 3>(norm, f * norm, 0.0f);
	}

	return (1.0f / sqrt(norm)) * result;
}

/**
 * Based on implementation for paper: 'Efficient numerical diagonalization of hermitian 3x3 matrices' by Joachim Kopp
 **/
//simple_vector<simple_vector<float, 3>, 3>
CUGIP_DECL_HYBRID matrix<float, 3, 3>
eigen_vectors(const symmetric_tensor<float, 3> &aTensor, const simple_vector<float, 3> &aEigenValues)
{
	matrix<float, 3, 3> result;

	auto maxEigenValue = max(abs(aEigenValues));
	auto threshold = 8.0f * EPSILON * maxEigenValue;

	result.column(0) = eigen_vector<0, 1>(aEigenValues[0], aTensor, sqr(threshold));
	if (abs(aEigenValues[0] - aEigenValues[1]) > threshold) {
		result.column(1) = eigen_vector<0, 1>(aEigenValues[1], aTensor, sqr(threshold));
	} else {
		// For degenerate eigenvalue, calculate second eigenvector according to
		//   v[1] = v[0] x (A - w[1]).e[i]
		int i;
		for (i = 0; i < 3; ++i) {
			auto v = aTensor.column(i);
			auto n0 = squared_magnitude(v);
			if (n0 > sqr(threshold)) {
				result.column(1) = cross(result.column(0), v);
				auto norm = squared_magnitude(result.column(1));
				// Accept cross product only if the angle between
				// the two vectors was not too small
				if (norm > sqr(256.0f * EPSILON) * n0) {
					norm = sqrt(1.0f / norm);
					result.column(1) = norm * result.column(1);
					break;
				}
			}
		}
		if (i == 3) {
			result.column(1) = find_orthonormal(result.column(0));
		}
	}

	result.column(2) = cross(result.column(0), result.column(1));
	return result;
}

CUGIP_DECL_HYBRID symmetric_tensor<float, 3>
matrix_from_eigen_vectors_and_values(const matrix<float, 3, 3> &aEigenVectors, const simple_vector<float, 3> &aEigenValues)
{
	symmetric_tensor<float, 3> result;

	// E * D * E'
	for (int i = 0; i < 3; ++i) {
		for (int j = i; j < 3; ++j) {
			simple_vector<float, 3> tmp = product(aEigenVectors.row(i), aEigenValues);
			result.get(i, j) = dot(tmp, aEigenVectors.row(j));
		}
	}
	return result;
}

/**
 * @}
 **/

}  // namespace cugip
