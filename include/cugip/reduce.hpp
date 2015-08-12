#pragma once

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cugip/image_view.hpp>
#include <cugip/access_utils.hpp>

namespace cugip {

/// Implements parallel reduction - application of associative operator on all image elements.
/// For example to sum all values in integer image:
/// \code
/// 	sum = Reduce(view, 0, thrust::plus<int>());
/// \endcode
/// \param view Processed image view - only constant element access needed.
/// \param initial_value Value used for result initialization (0 for sums, 1 for products, etc.)
/// \param reduction_operator Associative operator - it must be callable on device like this: result = reduction_operator(val1, val2).
/// \return Result of the operation.
template<typename TView, typename TOutputValue, typename TOperator>
TOutputValue Reduce(TView view, TOutputValue initial_value, TOperator reduction_operator);


/// Based on <a href="https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf">Optimizing Parallel Reduction in CUDA by Mark Harris</a>
template <typename TView, typename TOperator, typename TOutputValue, int tBlockSize>
CUGIP_GLOBAL void reduceKernel(
	TView view,
	TOutputValue initial_value,
	TOperator reduction_operator,
	TOutputValue *output
	)
{
	static_assert(tBlockSize >= 64, "Block must have at least 64 threads.");
	__shared__ TOutputValue sdata[tBlockSize];
	int tid = threadIdx.x;
	int index = blockIdx.x * (tBlockSize * 2) + tid;
	int grid_size = tBlockSize * 2 * gridDim.x;
	sdata[tid] = initial_value;
	int element_count = elementCount(view);
	while (index < element_count - tBlockSize) {
		sdata[tid] = reduction_operator(sdata[tid], reduction_operator(linear_access<TView>(view, index), linear_access<TView>(view, index + tBlockSize)));
		index += grid_size;
	}
	__syncthreads();
	if (index < element_count) {
		sdata[tid] = reduction_operator(sdata[tid], linear_access(view, index));
	}
	__syncthreads();
	if (tBlockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (tBlockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (tBlockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 64]);
		}
	}
	__syncthreads();
	for (int i = 32; i > 0; i >>= 1) {
		if (tid < i) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + i]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

template<typename TView, typename TOutputValue, typename TOperator>
TOutputValue reduce(TView view, TOutputValue initial_value, TOperator reduction_operator) {
	constexpr int kBucketSize = 4;  // Bundle more computation in one block - prevents thread idling. TODO(johny) - should be specified by call policy.
	dim3 block(512, 1, 1);
	dim3 grid(1 + (elementCount(view) - 1) / (block.x * kBucketSize), 1, 1);

	thrust::device_vector<TOutputValue> tmp_vector(grid.x);

	reduceKernel<TView, TOperator, TOutputValue, 512><<<grid, block>>>(view, initial_value, reduction_operator, tmp_vector.data().get());
	CUGIP_CHECK_ERROR_STATE("After ReduceKernel");
	CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	return thrust::reduce(tmp_vector.begin(), tmp_vector.end(), initial_value, reduction_operator);
}

}  // namespace cugip
