#pragma once

#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

CUGIP_DECL_HYBRID simple_vector<float, 3>
eigen_values(symmetric_tensor<float, 3> &aTensor)
{
	double m, c1, c0;

	// Determine coefficients of characteristic poynomial. We write
	//       | a   d   f  |
	//  A =  | d*  b   e  |
	//       | f*  e*  c  |
	double de = aTensor.get<0, 1>() * aTensor.get<1, 2>();                                    // d * e
	double dd = sqr(aTensor.get<0, 1>());                                         // d^2
	double ee = sqr(aTensor.get<1, 2>());                                         // e^2
	double ff = sqr(aTensor.get<0, 2>());                                         // f^2
	m  = matrix_trace(aTensor);//A[0][0] + A[1][1] + A[2][2];
	c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2]) - (dd + ee + ff);       // a*b + a*c + b*c - d^2 - e^2 - f^2
	c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2] - 2.0 * A[0][2]*de; // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

	double p, sqrt_p, q, c, s, phi;
	p = sqr(m) - 3.0 * c1;
	q = m*(p - (3.0/2.0) * c1) - (27.0/2.0) * c0;
	sqrt_p = sqrt(fabs(p));

	phi = 27.0 * (0.25 * sqr(c1) * (p - c1) + c0*(q + 27.0/4.0 * c0));
	phi = (1.0 / 3.0) * atan2(sqrt(fabs(phi)), q);

	c = sqrt_p*cos(phi);
	s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

	w[1]  = (1.0/3.0)*(m - c);
	w[2]  = w[1] + s;
	w[0]  = w[1] + c;
	w[1] -= s;
}

// ----------------------------------------------------------------------------
int
eigen_vectors(symmetric_tensor<float, 3> &aTensor, double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//     (according to the documentation, only the upper triangular part needs
//     to be filled)
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
	simple_vector<float, 3> eigenValues;
	auto maxEigenValue = max(abs(eigenValues)); // The eigenvalue of maximum modulus

	double norm;          // Squared norm or inverse norm of current eigenvector
	double n0, n1;        // Norm of first and second columns of A
	double n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
	double thresh;        // Small number used as threshold for floating point comparisons
	double error;         // Estimated maximum roundoff error in some steps
	double f, t;          // Intermediate storage
	int i, j;             // Loop counters

	thresh = sqr(8.0 * DBL_EPSILON * wmax);

	// Prepare calculation of eigenvectors
	n0tmp   = sqr(A[0][1]) + sqr(A[0][2]);
	n1tmp   = sqr(A[0][1]) + sqr(A[1][2]);
	Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
	Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
	Q[2][1] = sqr(A[0][1]);

	// Calculate first eigenvector by the formula
	//   v[0] = (A - w[0]).e1 x (A - w[0]).e2
	A[0][0] -= w[0];
	A[1][1] -= w[0];
	Q[0][0] = Q[0][1] + A[0][2]*w[0];
	Q[1][0] = Q[1][1] + A[1][2]*w[0];
	Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
	norm    = sqr(Q[0][0]) + sqr(Q[1][0]) + sqr(Q[2][0]);
	n0      = n0tmp + sqr(A[0][0]);
	n1      = n1tmp + sqr(A[1][1]);
	error   = n0 * n1;

	if (n0 <= thresh) {         // If the first column is zero, then (1,0,0) is an eigenvector
		Q[0][0] = 1.0;
		Q[1][0] = 0.0;
		Q[2][0] = 0.0;
	} else if (n1 <= thresh) {   // If the second column is zero, then (0,1,0) is an eigenvector
		Q[0][0] = 0.0;
		Q[1][0] = 1.0;
		Q[2][0] = 0.0;
	} else if (norm < sqr(64.0 * DBL_EPSILON) * error) {
		// If angle between A[0] and A[1] is too small, don't use
		t = sqr(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
		f = -A[0][0] / A[0][1];
		if (sqr(A[1][1]) > t) {
			t = sqr(A[1][1]);
			f = -A[0][1] / A[1][1];
		}
		if (sqr(A[1][2]) > t) {
			f = -A[0][2] / A[1][2];
		}
		norm    = 1.0/sqrt(1 + sqr(f));
		Q[0][0] = norm;
		Q[1][0] = f * norm;
		Q[2][0] = 0.0;
	} else { // This is the standard branch
		norm = sqrt(1.0 / norm);
		for (j=0; j < 3; j++) {
			Q[j][0] = Q[j][0] * norm;
		}
	}


	// Prepare calculation of second eigenvector
	t = w[0] - w[1];
	if (fabs(t) > 8.0 * DBL_EPSILON * wmax) {
		// For non-degenerate eigenvalue, calculate second eigenvector by the formula
		//   v[1] = (A - w[1]).e1 x (A - w[1]).e2
		A[0][0] += t;
		A[1][1] += t;
		Q[0][1]  = Q[0][1] + A[0][2]*w[1];
		Q[1][1]  = Q[1][1] + A[1][2]*w[1];
		Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
		norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
		n0       = n0tmp + SQR(A[0][0]);
		n1       = n1tmp + SQR(A[1][1]);
		error    = n0 * n1;

		if (n0 <= thresh) {       // If the first column is zero, then (1,0,0) is an eigenvector
			Q[0][1] = 1.0;
			Q[1][1] = 0.0;
			Q[2][1] = 0.0;
		} else if (n1 <= thresh) { // If the second column is zero, then (0,1,0) is an eigenvector
			Q[0][1] = 0.0;
			Q[1][1] = 1.0;
			Q[2][1] = 0.0;
		} else if (norm < SQR(64.0 * DBL_EPSILON) * error) {
			// If angle between A[0] and A[1] is too small, don't use
			t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
			f = -A[0][0] / A[0][1];
			if (SQR(A[1][1]) > t) {
				t = SQR(A[1][1]);
				f = -A[0][1] / A[1][1];
			}
			if (SQR(A[1][2]) > t) {
				f = -A[0][2] / A[1][2];
			}
			norm    = 1.0/sqrt(1 + SQR(f));
			Q[0][1] = norm;
			Q[1][1] = f * norm;
			Q[2][1] = 0.0;
		} else {
			norm = sqrt(1.0 / norm);
			for (j=0; j < 3; j++) {
				Q[j][1] = Q[j][1] * norm;
			}
		}
	} else {
		// For degenerate eigenvalue, calculate second eigenvector according to
		//   v[1] = v[0] x (A - w[1]).e[i]
		//
		// This would really get to complicated if we could not assume all of A to
		// contain meaningful values.
		A[1][0]  = A[0][1];
		A[2][0]  = A[0][2];
		A[2][1]  = A[1][2];
		A[0][0] += w[0];
		A[1][1] += w[0];
		for (i=0; i < 3; i++) {
			A[i][i] -= w[1];
			n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
			if (n0 > thresh) {
				Q[0][1]  = Q[1][0]*A[2][i] - Q[2][0]*A[1][i];
				Q[1][1]  = Q[2][0]*A[0][i] - Q[0][0]*A[2][i];
				Q[2][1]  = Q[0][0]*A[1][i] - Q[1][0]*A[0][i];
				norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
				if (norm > SQR(256.0 * DBL_EPSILON) * n0) {
					// Accept cross product only if the angle between
					// the two vectors was not too small
					norm = sqrt(1.0 / norm);
					for (j=0; j < 3; j++)
					Q[j][1] = Q[j][1] * norm;
					break;
				}
			}
		}

		if (i == 3) {    // This means that any vector orthogonal to v[0] is an EV.
			for (j=0; j < 3; j++) {
				if (Q[j][0] != 0.0) {
					// Find nonzero element of v[0] ...
					// ... and swap it with the next one
					norm          = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j+1)%3][0]));
					Q[j][1]       = Q[(j+1)%3][0] * norm;
					Q[(j+1)%3][1] = -Q[j][0] * norm;
					Q[(j+2)%3][1] = 0.0;
					break;
				}
			}
		}
	}


	// Calculate third eigenvector according to
	//   v[2] = v[0] x v[1]
	Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
	Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
	Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];

	return 0;
}
