#pragma once

#include <limits>
#include <cugip/functors.hpp>
#include <cugip/view_arithmetics.hpp>
#include <cugip/reduce.hpp>

namespace cugip {



///https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
template <typename TView, typename TOutputView, typename TOutputValue, typename TOperator, int tDirection, int tBlockSize>
CUGIP_GLOBAL void prescan(
		TView aInputView,
		TOutputView aOutputView,
		TOutputValue aInitialValue,
		TOperator aReductionOperator)
{
	using Indexing = index_traits<dimension<TView>::value, dimension<TView>::value, tDirection, tBlockSize>;
	__shared__ TOutputValue sdata[2*tBlockSize];
	__syncthreads();  // Wait for all threads to call constructor on sdata
	//SharedMemoryStaticArray<TOutputValue, tBlockSize, true> sdata;

	int tid = threadIdx.x;
	int count_in_dimension = aInputView.dimensions()[tDirection];
	sdata[tid] = aInitialValue;
	sdata[tid + tBlockSize] = aInitialValue;
	__syncthreads();
	int n = 2 * tBlockSize; //TODO

	auto index = Indexing::GetInputIndex();

	int offset = 1;

	auto current_index = index;
	auto sdata_index = tid;
	while (sdata_index < n && current_index[tDirection] < count_in_dimension) {
		sdata[sdata_index] = aInputView[current_index]; // load input into shared memory
		current_index[tDirection] += tBlockSize;
		sdata_index += tBlockSize;
	}

	for (int d = n>>1; d > 0; d >>= 1) {                    // build sum in place up the tree
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			sdata[bi] = aReductionOperator(sdata[ai], sdata[bi]);
		}
		offset *= 2;
	}
	if (tid == 0) { sdata[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			auto t = sdata[ai];
			sdata[ai] = sdata[bi];
			sdata[bi] = aReductionOperator(t, sdata[bi]);
		}
	}
	__syncthreads();

	current_index = index;
	sdata_index = tid;
	while (sdata_index < n && current_index[tDirection] < count_in_dimension) {
		aOutputView[current_index] = sdata[sdata_index];
		current_index[tDirection] += tBlockSize;
		sdata_index += tBlockSize;
	}
}

template <typename TView, typename TOutputView, typename TOutputValue, typename TOperator, int tDirection, int tBlockSize>
CUGIP_GLOBAL void scan_finalization(
		TView aInputView,
		TOutputView aOutputView,
		TOutputValue aInitialValue,
		TOperator aReductionOperator)
{
	using Indexing = index_traits<dimension<TOutputView>::value, dimension<TOutputView>::value, tDirection, tBlockSize / 2>;

	__shared__ typename TOutputView::value_type prefix;

	int tid = threadIdx.x;
	int count_in_dimension = aOutputView.dimensions()[tDirection];

	if (tid == 0) {
		prefix = aInitialValue;
	}
	__syncthreads();

	auto index = Indexing::GetInputIndex();


	for (int i = 1; i < (count_in_dimension + tBlockSize - 1) / tBlockSize; ++i) {
		if (tid == 0) {
			index[tDirection] = i * tBlockSize - 1;
			prefix = aReductionOperator(aReductionOperator(prefix, aOutputView[index]), aInputView[index]); // TODO better solution for prefix sum
		}
		__syncthreads();

		index[tDirection] = i * tBlockSize + tid;
		if (index[tDirection] <  count_in_dimension) {
			aOutputView[index] = aReductionOperator(aOutputView[index], prefix);
		}
		__syncthreads();
	}
}



template<bool tIsOnDevice>
struct scan_implementation {
	//TODO host version of scan algorithm
	template<typename TInputView, typename TOutputView, typename TOutputValue, typename TOperator, int tDirection, typename TExecutionConfig>
	static void
	run(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOutputValue aInitialValue, TOperator aOperator, TExecutionConfig aExecutionConfig)
	{
		constexpr int cDimension = dimension<TInputView>::value;
		auto size = aInput.dimensions();
		auto reduced_size = remove_dimension(size, tDirection);
		constexpr int cBlockSize = 512;

		int block_count = 1 + (size[tDirection] - 1) / cBlockSize;

		dim3 block(cBlockSize, 1, 1);
		dim3 grid(block_count, reduced_size[0], cDimension > 2 ? reduced_size[1] : 1);
		prescan<TInputView, TOutputView, TOutputValue, TOperator, tDirection, cBlockSize>
			<<<grid, block, 0, aExecutionConfig.stream>>>(aInput, aOutput, aInitialValue, aOperator);

		if (size[tDirection] > 2 * cBlockSize) {
			dim3 block(2 * cBlockSize, 1, 1);
			dim3 grid(1, reduced_size[0], cDimension > 2 ? reduced_size[1] : 1);
			scan_finalization<TInputView, TOutputView, TOutputValue, TOperator, tDirection, cBlockSize * 2>
				<<<grid, block, 0, aExecutionConfig.stream>>>(aInput, aOutput, aInitialValue, aOperator);
		}
	}
};

/** \addtogroup meta_algorithm
 * @{
 **/


template<typename TInputView, typename TOutputView, typename TOutputValue, typename TOperator, int tDirection, typename TExecutionConfig>
void scan(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOutputValue aInitialValue, IntValue<tDirection> aDirection, TOperator aOperator, TExecutionConfig aExecutionConfig) {

	scan_implementation<true>::run<TInputView, TOutputView, TOutputValue, TOperator, tDirection, TExecutionConfig>(
		aInput, aOutput, aTmpView, aInitialValue, aOperator, aExecutionConfig);
}

template<typename TInputView, typename TOutputView, typename TOutputValue, typename TOperator, int tDirection>
void scan(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOutputValue aInitialValue, IntValue<tDirection> aDirection, TOperator aOperator) {
	scan(aInput, aOutput, aTmpView, aInitialValue, aDirection, aOperator, ScanExecutionConfig{});
}

/**
 * @}
 **/

} //namespace cugip

