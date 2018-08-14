#pragma once

#include <limits>
#include <cugip/functors.hpp>
#include <cugip/view_arithmetics.hpp>
#include <cugip/image.hpp>

#include <thrust/device_vector.h>

namespace cugip {


template <int tInputDimension, int tOutputDimension, int tDirection, int tBlockSize>
struct index_traits;

template <int tDirection, int tBlockSize>
struct index_traits<2, 1, tDirection, tBlockSize> {
	CUGIP_DECL_DEVICE
	static vect2i_t GetInputIndex() {
		auto tmp = simple_vector<int, 1>(blockIdx.y);
		return insert_dimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	CUGIP_DECL_DEVICE
	static int GetOutputIndex(vect2i_t index) {
		return remove_dimension(index, tDirection)[0];
	}
};

template <int tDirection, int tBlockSize>
struct index_traits<2, 2, tDirection, tBlockSize> {
	CUGIP_DECL_DEVICE
	static vect2i_t GetInputIndex() {
		auto tmp = simple_vector<int, 1>(blockIdx.y);
		return insert_dimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	CUGIP_DECL_DEVICE
	static vect2i_t GetOutputIndex(vect2i_t index) {
		auto tmp = simple_vector<int, 1>(blockIdx.y);
		return insert_dimension(tmp, int(blockIdx.x), tDirection);
	}
};

template <int tDirection, int tBlockSize>
struct index_traits<3, 2, tDirection, tBlockSize> {
	CUGIP_DECL_DEVICE
	static vect3i_t GetInputIndex() {
		auto tmp = vect2i_t(blockIdx.y, blockIdx.z);
		return insert_dimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	CUGIP_DECL_DEVICE
	static vect2i_t GetOutputIndex(vect3i_t index) {
		return remove_dimension(index, tDirection);
	}
};

template <int tDirection, int tBlockSize>
struct index_traits<3, 3, tDirection, tBlockSize> {
	CUGIP_DECL_DEVICE
	static vect3i_t GetInputIndex() {
		auto tmp = vect2i_t(blockIdx.y, blockIdx.z);
		return insert_dimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	CUGIP_DECL_DEVICE
	static vect3i_t GetOutputIndex(vect3i_t index) {
		auto tmp = vect2i_t(blockIdx.y, blockIdx.z);
		return insert_dimension(tmp, int(blockIdx.x), tDirection);
	}
};


/// Based on <a href="https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf">Optimizing Parallel Reduction in CUDA by Mark Harris</a>
template <typename TView, typename TOutputView, typename TOutputValue, typename TOperator, int tDimension, int tBlockSize>
CUGIP_GLOBAL void dimensionReduceKernel(
	TView view,
	TOutputView output_view,
	TOutputValue initial_value,
	TOperator reduction_operator)
{
	using Indexing = index_traits<TView::kDimension, TOutputView::kDimension, tDimension, tBlockSize>;
	__shared__ TOutputValue sdata[tBlockSize];
	__syncthreads();  // Wait for all threads to call constructor on sdata
	//SharedMemoryStaticArray<TOutputValue, tBlockSize, true> sdata;

	int tid = threadIdx.x;
	int count_in_dimension = view.dimensions()[tDimension];
	sdata[tid] = initial_value;

	auto index = Indexing::GetInputIndex();
	auto output_index = Indexing::GetOutputIndex(index);
	int step = tBlockSize * 2 * gridDim.x;

	while (index[tDimension] < count_in_dimension - tBlockSize) {
		auto other_index = index;
		other_index[tDimension] += tBlockSize;
		sdata[tid] = reduction_operator(sdata[tid], reduction_operator(view[index], view[other_index]));
		index[tDimension] += step;
	}
	__syncthreads();
	if (index[tDimension] < count_in_dimension) {
		sdata[tid] = reduction_operator(sdata[tid], view[index]);
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
		output_view[output_index] = sdata[0];
	}
}


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

template<bool tOnDevice>
struct DimensionReduceImplementation
{
	template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator, typename TExecutionPolicy>
	static void run(TView view, TOutputView output_view, IntValue<tDimension> direction, TOutputValue initial_value, TOperator reduction_operator, TExecutionPolicy execution_policy) {
		auto size = view.dimensions();
		auto reduced_size = remove_dimension(size, tDimension);

		if (reduced_size != output_view.dimensions()) {
			//TODO CUGIP_THROW(IncompatibleViewSizes() << GetViewPairSizesErrorInfo(reduced_size, output_view.Size()));
		}

		constexpr int cBlockSize = 256;
		constexpr int cBucketSize = 2;
		int block_count = 1 + (size[tDimension] - 1) / (cBlockSize * cBucketSize);
		// TODO - better setup of bucket size and block size depending on input size
		if (block_count > 1) {
			dim3 block(cBlockSize, 1, 1);
			dim3 grid(block_count, reduced_size[0], reduced_size.dim >= 2 ? reduced_size[1] : 1);

			auto tmp_size = size;
			tmp_size[tDimension] = block_count;
			device_image<TOutputValue, dimension<TView>::value> tmp_buffer(tmp_size);

			dimensionReduceKernel<TView, decltype(view(tmp_buffer)), TOutputValue, TOperator, tDimension, cBlockSize><<<grid, block, 0, execution_policy.cuda_stream>>>(view, view(tmp_buffer), initial_value, reduction_operator);
			//CUGIP_CHECK_ERROR_AFTER_KERNEL("DimensionReduceKernel tmp buffer", grid, block);
			run(tmp_buffer.ConstView(), output_view, direction, initial_value, reduction_operator, execution_policy);
		} else {
			dim3 block(cBlockSize, 1, 1);
			dim3 grid(1 + (size[tDimension] - 1) / block.x, reduced_size[0], reduced_size.dim >= 2 ? reduced_size[1] : 1);

			dimensionReduceKernel<TView, TOutputView, TOutputValue, TOperator, tDimension, cBlockSize><<<grid, block, 0, execution_policy.cuda_stream>>>(view, output_view, initial_value, reduction_operator);
			//CUGIP_CHECK_ERROR_AFTER_KERNEL("DimensionReduceKernel", grid, block);
			CUGIP_CHECK(cudaStreamSynchronize(execution_policy.cuda_stream));
		}
	}

};

template<>
struct DimensionReduceImplementation<false>
{
	/*template<typename TView, typename TOutputValue, typename TOperator>
	static TOutputValue
	run(TView view, TOutputValue initial_value, TOperator reduction_operator) {
		TOutputValue result = initial_value;
		for (int64_t i = 0; i < elementCount(view); ++i) {
			result = reduction_operator(result, linear_access(view, i));
		}
		return result;
	}*/
};


template<bool tOnDevice>
struct ReduceImplementation
{
	template<typename TView, typename TOutputValue, typename TOperator>
	static TOutputValue
	run(TView view, TOutputValue initial_value, TOperator reduction_operator) {
		constexpr int kBucketSize = 4;  // Bundle more computation in one block - prevents thread idling. TODO(johny) - should be specified by call policy.
		dim3 block(512, 1, 1);
		dim3 grid(1 + (elementCount(view) - 1) / (block.x * kBucketSize), 1, 1);

		thrust::device_vector<TOutputValue> tmp_vector(grid.x);

		reduceKernel<TView, TOperator, TOutputValue, 512><<<grid, block>>>(view, initial_value, reduction_operator, tmp_vector.data().get());
		CUGIP_CHECK_ERROR_STATE("After ReduceKernel");
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
		return thrust::reduce(tmp_vector.begin(), tmp_vector.end(), initial_value, reduction_operator);
	}

};

template<>
struct ReduceImplementation<false>
{
	template<typename TView, typename TOutputValue, typename TOperator>
	static TOutputValue
	run(TView view, TOutputValue initial_value, TOperator reduction_operator) {
		TOutputValue result = initial_value;
		for (int64_t i = 0; i < elementCount(view); ++i) {
			result = reduction_operator(result, linear_access(view, i));
		}
		return result;
	}
};

template<typename TView, typename TOutputValue, typename TOperator>
TOutputValue reduce(TView view, TOutputValue initial_value, TOperator reduction_operator) {
	return ReduceImplementation<is_device_view<TView>::value>::run(view, initial_value, reduction_operator);
}

template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator, typename TExecutionPolicy>
void dimension_reduce(TView view, TOutputView output_view, IntValue<tDimension> dimension, TOutputValue initial_value, TOperator reduction_operator, TExecutionPolicy execution_policy)
{
	DimensionReduceImplementation<is_device_view<TView>::value>::run(view, output_view, dimension, initial_value, reduction_operator, execution_policy);
}


template<typename TView, class>
typename TView::value_type min(TView view)
{
	using std::numeric_limits;
	return reduce(view, numeric_limits<typename TView::value_type>::max(), MinFunctor());
}

template<typename TView, class>
typename TView::value_type max(TView view)
{
	using std::numeric_limits;
	return reduce(view, numeric_limits<typename TView::value_type>::lowest(), MaxFunctor());
}

template<typename TView, typename TOutputValue, class>
TOutputValue sum(TView view, TOutputValue initial_value)
{
	using std::numeric_limits;
	return reduce(view, initial_value, SumValuesFunctor());
}

template<typename TView, class>
typename TView::value_type sum(TView view)
{
	return reduce(view, typename TView::value_type(0), SumValuesFunctor());
}

template<typename TView1, typename TView2, typename TOutputValue, class>
TOutputValue sum_differences(TView1 view1, TView2 view2, TOutputValue initial_value)
{
	//return 0;
	return sum(abs_view(subtract(view1, view2)), initial_value);
}

}  // namespace cugip
