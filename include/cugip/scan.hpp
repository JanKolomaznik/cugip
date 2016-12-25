#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>
#include <cugip/image_locator.hpp>
#include <cugip/math.hpp>

namespace cugip {

#define SCAN_BLOCK_SIZE 256

namespace detail {

template<typename TInputView, typename TOutputView, typename TOperator, int tDim, int tBlockSize, bool tStoreIntermediateR>
CUGIP_GLOBAL void
kernel_scan_block_dim(TInputView aInput, TOutputView aOutput, TOutputView aIntermediateResultsView, TOperator aOperator)
{
	typedef typename TOutputView::value_type value_type;
	typedef typename TOutputView::coord_t coord_t;
	typedef typename TOutputView::extents_t extents_t;
	CUGIP_SHARED value_type tmpData[tBlockSize+1];
	int offset = tBlockSize >> 1;

	coord_t threadCoord(threadIdx.x, threadIdx.y, threadIdx.z);
	coord_t processedBlockDim(blockDim.x, blockDim.y, blockDim.z);
	get<tDim>(processedBlockDim) = tBlockSize;

	int thid = get<tDim>(threadCoord);
	int bufferIdx1 = get<tDim>(threadCoord);
	int bufferIdx2 = bufferIdx1 + offset;
	coord_t coord1 = coord_t(blockIdx.x * processedBlockDim[0] + threadIdx.x, blockIdx.y * processedBlockDim[1] + threadIdx.y, blockIdx.z * processedBlockDim[2] + threadIdx.z);
	coord_t coord2 = coord1;
	get<tDim>(coord2) += offset;
	extents_t extents = aInput.dimensions();

	if (cugip::less(coord1, extents)) {
		tmpData[bufferIdx1] = aInput[coord1];
	} else {
		tmpData[bufferIdx1] = 0; //TODO use right mask for background
	}
	if (cugip::less(coord2, extents)) {
		tmpData[bufferIdx2] = aInput[coord2];
	} else {
		tmpData[bufferIdx2] = 0; //TODO use right mask for background
	}
	offset = 1;
	for (int d = tBlockSize >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (thid < d) {
			int ai = offset*((thid << 1) + 1) - 1;
			int bi = offset*((thid << 1) + 2) - 1;
			tmpData[bi] = aOperator(tmpData[ai], tmpData[bi]);
		}
		offset *= 2;
	}
	if (thid == 0) {
		tmpData[tBlockSize] = tmpData[tBlockSize-1];
		tmpData[tBlockSize-1] = 0;
		if (tBlockSize < get<tDim>(extents)) {
//		printf("ll\n");
			coord_t blockSumCoord(blockIdx.x, blockIdx.y, blockIdx.z);
			if (get<tDim>(blockSumCoord) == 0) {
				aIntermediateResultsView[blockSumCoord] = 0;
			}
			++get<tDim>(blockSumCoord);
			aIntermediateResultsView[blockSumCoord] = tmpData[tBlockSize];
		}
	} // clear the last element



	for (int d = 1; d < tBlockSize; d *= 2) {// traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (thid < d) {

			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			value_type t = tmpData[ai];
			tmpData[ai] = tmpData[bi];
			tmpData[bi] = aOperator(tmpData[bi], t);
		}
	}
	__syncthreads();
	if (cugip::less(coord1, extents)) {
		aOutput[coord1] = tmpData[bufferIdx1 + 1];
	}
	if (cugip::less(coord2, extents)) {
		aOutput[coord2] = tmpData[bufferIdx2 + 1];
	}
}

template<typename TOutputView, typename TOperator, int tDim, int tBlockSize>
CUGIP_GLOBAL void
kernel_update_block_dim(TOutputView aOutput, TOutputView aIntermediateResultsView, TOperator aOperator)
{
	typedef typename TOutputView::value_type value_type;
	typedef typename TOutputView::coord_t coord_t;
	typedef typename TOutputView::extents_t extents_t;

	int offset = tBlockSize >> 1;

	coord_t threadCoord(threadIdx.x, threadIdx.y, threadIdx.z);
	coord_t processedBlockDim(blockDim.x, blockDim.y, blockDim.z);
	get<tDim>(processedBlockDim) = tBlockSize;

	int thid = get<tDim>(threadCoord);
	coord_t coord1 = coord_t(blockIdx.x * processedBlockDim[0] + threadIdx.x, blockIdx.y * processedBlockDim[1] + threadIdx.y, blockIdx.z * processedBlockDim[2] + threadIdx.z);
	coord_t coord2 = coord1;
	get<tDim>(coord2) += offset;
	coord_t blockSumCoord(blockIdx.x, blockIdx.y, blockIdx.z);
	CUGIP_SHARED value_type intermediateValue;
       	if (thid == 0) {
		intermediateValue = aIntermediateResultsView[blockSumCoord];
		//printf("%dx%d -  %d\n", (int)blockIdx.x, (int)blockIdx.y, (int)intermediateValue);
	}
	__syncthreads();

	extents_t extents = aOutput.dimensions();

	if (cugip::less(coord1, extents)) {
		aOutput[coord1] = aOperator(intermediateValue, aOutput[coord1]);
	}
	if (cugip::less(coord2, extents)) {
		aOutput[coord2] = aOperator(intermediateValue, aOutput[coord2]);
	}
}

template<typename TInputView, typename TOutputView, typename TOperator, int tDim>
void
scan_block_dim(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOperator aOperator)
{
	//TODO generic method for thread distribution over the data
	dim3 threadBlockSize(1, 1, 1);
	get<tDim>(threadBlockSize) = SCAN_BLOCK_SIZE;
	dim3 processedBlockSize(1, 1, 1);
	get<tDim>(processedBlockSize) = 2*SCAN_BLOCK_SIZE;
	dim3 gridSize((aInput.dimensions().template get<0>() / processedBlockSize.x + 1),
			aInput.dimensions().template get<1>() / processedBlockSize.y + 1, 1);

	D_PRINT("Executing kernel: threadBlockSize = "
	               << threadBlockSize
	               << "; gridSize = "
	               << gridSize
	       );

	detail::kernel_scan_block_dim<
				TInputView,
				TOutputView,
				TOperator,
				tDim,
				SCAN_BLOCK_SIZE*2
			>
		<<<gridSize, threadBlockSize>>>(aInput, aOutput, aTmpView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_scan_block_dim");
}

template<typename TOutputView, typename TOperator, int tDim>
void
update_block_dim(TOutputView aOutput, TOutputView aTmpView, TOperator aOperator)
{
	//TODO generic method for thread distribution over the data
	dim3 threadBlockSize(1, 1, 1);
	get<tDim>(threadBlockSize) = SCAN_BLOCK_SIZE;
	dim3 processedBlockSize(1, 1, 1);
	get<tDim>(processedBlockSize) = 2*SCAN_BLOCK_SIZE;
	dim3 gridSize((aOutput.dimensions().template get<0>() / processedBlockSize.x + 1),
			aOutput.dimensions().template get<1>() / processedBlockSize.y + 1, 1);

	D_PRINT("Executing kernel: threadBlockSize = "
	               << threadBlockSize
	               << "; gridSize = "
	               << gridSize
	       );

	detail::kernel_update_block_dim<
				TOutputView,
				TOperator,
				tDim,
				SCAN_BLOCK_SIZE*2
			>
		<<<gridSize, threadBlockSize>>>(aOutput, aTmpView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_update_block_dim");
}


template<typename TInputView, typename TOutputView, typename TOperator, int tDim>
void
scan_dim(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOperator aOperator)
{
	TOutputView intermediateResults = aTmpView;
	//TOutputView intermediateResults = sub_image_view(aTmpView, typename TOutputView::coord_t(), TOutputView::extents_t(adafaf));
	detail::scan_block_dim<
				TInputView,
				TOutputView,
				TOperator,
				tDim
			>(aInput, aOutput, intermediateResults, aOperator);

	CUGIP_ASSERT(get<tDim>(aInput.dimensions()) < (4*SCAN_BLOCK_SIZE*SCAN_BLOCK_SIZE) );
	//TODO handle intermediate results - proper recursive call
	scan_dim<
		TOutputView,
		TOutputView,
		TOperator,
		tDim
		>(intermediateResults, intermediateResults, intermediateResults, aOperator);

	detail::update_block_dim<
				TOutputView,
				TOperator,
				tDim
			>(aOutput, intermediateResults, aOperator);
}


} //namespace detail

/** \addtogroup meta_algorithm
 * @{
 **/

template<typename TInputView, typename TOutputView, typename TOperator>
void
scan(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOperator aOperator)
{
	detail::scan_dim<
				TInputView,
				TOutputView,
				TOperator,
				0
			//>(aInput, aTmpView, aOutput, aOperator);
			>(aInput, aOutput, aTmpView, aOperator);

/*	detail::scan_dim<
				TOutputView,
				TOutputView,
				TOperator,
				1
			>(aTmpView, aOutput, aTmpView, aOperator);*/
}

/**
 * @}
 **/


} //namespace cugip
