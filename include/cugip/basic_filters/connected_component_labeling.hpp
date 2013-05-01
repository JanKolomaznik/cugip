#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/image_view.hpp>
#include <cugip/access_utils.hpp>

namespace cugip {

namespace detail {

//-----------------------------------------------------------------------------
template <typename TImageView, typename TLUTBufferView>
CUGIP_GLOBAL void 
init_lut_kernel(TImageView aImageView, TLUTBufferView aLut)
{ 
	int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if (idx < multiply(aImageView.dimensions())) {
		//TODO - check 0
		aLut[idx+1] = linear_access(aImageView, idx);// = outBuffer.mData[idx] != 0 ? idx+1 : 0;
	}
}

template <typename TImageView, typename TLUTBufferView>
void
init_lut(TImageView aImageView, TLUTBufferView aLut)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	init_lut_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLut);
}

//-----------------------------------------------------------------------------
template <typename TLUTBufferView>
CUGIP_GLOBAL void
update_lut_kernel(TLUTBufferView aLUT)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if (idx < aLUT.dimensions()) {
		typename TLUTBufferView::value_type ref = aLUT[idx];
		typename TLUTBufferView::value_type newref = ref;

		if (ref) {
			do {
				ref = newref;
				newref = aLUT[ref];
			} while (ref != newref);
			atomicExch(&aLUT[idx], newref);
		}
	}
}

template <typename TLUTBufferView>
void
update_lut(TLUTBufferView aLUT)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D((aLUT.dimensions() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	update_lut_kernel<<< gridSize1D, blockSize1D >>>(aLUT);
}

//-----------------------------------------------------------------------------
template <typename TImageView, typename TLUTBufferView>
CUGIP_GLOBAL void
update_labels_kernel(TImageView aImageView, TLUTBufferView aLUT)
{
	int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if (idx < multiply(aImageView.dimensions())) {
		uint64_t label = linear_access(aImageView, idx);
		if ( label > 0 ) {
			linear_access(aImageView, idx) = aLUT[label];
		}
	}
}

template <typename TImageView, typename TLUTBufferView>
void
update_labels(TImageView aImageView, TLUTBufferView aLUT)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	update_labels_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLUT);
}
//-----------------------------------------------------------------------------
template <typename TImageView, typename TLabelView>
CUGIP_GLOBAL void 
init_labels_kernel(TImageView aImageView, TLabelView aLabelView)
{ 
	int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if (idx < multiply(aImageView.dimensions())) {
		//TODO - check 0
		linear_access(aLabelView, idx) = (linear_access(aImageView, idx) != 0) ? idx+1 : 0;
	}
}

template <typename TImageView, typename TLabelView>
void
init_labels(TImageView aImageView, TLabelView aLabelView)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	init_labels_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLabelView);
}


//-----------------------------------------------------------------------------
template <typename TImageView, size_t tSharedMemoryBufferSize>
CUGIP_GLOBAL void
block_ccl_kernel(TImageView aImageView)
{
	typedef typename TImageView::value_type value_type;
	typedef typename TImageView::coord_t coord_t;
	typedef typename TImageView::extents_t extents_t;
	//TODO 3D version
	coord_t coord = coord_t(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	extents_t extents = aImageView.dimensions();

	CUGIP_ASSERT(tSharedMemoryBufferSize >= (blockDim.x * blockDim.y * blockDim.z));

	coord_t threadCoord(threadIdx.x, threadIdx.y/*, threadIdx.z*/);
	extents_t blockExtents(blockDim.x, blockDim.y/*blockDim.z*/);
	CUGIP_SHARED value_type blockData[tSharedMemoryBufferSize];
	device_image_view<value_type> blockView = cugip::view(device_ptr<value_type>(blockData), blockExtents);

	if (cugip::less(coord, extents)) {
		blockView[threadCoord] = aImageView[coord];
	} else {
		blockView[threadCoord] = 0; //TODO use right mask for background
	}
	__syncthreads();

	value_type current = blockView[threadCoord];
	int changed;
	do {
		//TODO more generic neighborhood processing
		changed = 0;
		if (current != 0) {
			coord_t coord2 = threadCoord;
			coord2[0] = max(coord2[0]-1, 0);
			value_type newValue = blockView[coord2];
			if ((newValue < current) && (newValue != 0)) {
				blockView[threadCoord] = current =  newValue;
				++changed;
			}
		}
		__syncthreads();
		if (current != 0) {
			coord_t coord2 = threadCoord;
			coord2[1] = max(coord2[1]-1, 0);
			value_type newValue = blockView[coord2];
			if ((newValue < current) && (newValue != 0)) {
				blockView[threadCoord] = current =  newValue;
				++changed;
			}
		}
		__syncthreads();
		if (current != 0) {
			coord_t coord2 = threadCoord;
			coord2[0] = min(coord2[0]+1, (int)blockExtents[0]-1);
			value_type newValue = blockView[coord2];
			if ((newValue < current) && (newValue != 0)) {
				blockView[threadCoord] = current =  newValue;
				++changed;
			}
		}
		__syncthreads();
		if (current != 0) {
			coord_t coord2 = threadCoord;
			coord2[1] = min(coord2[1]+1, (int)blockExtents[1]-1);
			value_type newValue = blockView[coord2];
			if ((newValue < current) && (newValue != 0)) {
				blockView[threadCoord] = current =  newValue;
				++changed;
			}
		}
	} while (__syncthreads_or(changed));

	// store results
	if (cugip::less(coord, extents)) {
		aImageView[coord] = blockView[threadCoord];
	}
}

template <typename TImageView>
void
block_ccl(TImageView aImageView)
{
	dim3 blockSize(16, 16, 1);
	dim3 gridSize(
			(aImageView.dimensions().template get<0>() / blockSize.x + 1), 
			(aImageView.dimensions().template get<1>() / blockSize.y + 1), 
			1);

	//TODO Fix shared memory buffer size template parameter settings
	block_ccl_kernel<TImageView, 16*16><<<gridSize, blockSize>>>(aImageView);
}

template <typename TLUT>
CUGIP_DECL_DEVICE void
addEquivalence(TLUT &aLut, typename TLUT::value_type aVal1, typename TLUT::value_type aVal2)
{
	typedef typename TLUT::value_type value_type;

	//atomicMin(&aLut[val1], m);

	value_type old = atomicMin(&aLut[aVal2], aVal1);
	if (old < aVal1) {
		addEquivalence(aLut, old, aVal1);
	} else if (old != aVal2) {
		addEquivalence(aLut, aVal1, old);
	}
}

template <typename TImageView, typename TLUT>
CUGIP_GLOBAL void
merge_ccl_blocks_kernel(TImageView aImageView, TLUT aLut)
{
	typedef typename TImageView::value_type value_type;
	typedef typename TImageView::coord_t coord_t;
	typedef typename TImageView::extents_t extents_t;

	//TODO 3D version
	value_type val1 = 0;
	value_type val2 = 0;
	extents_t extents = aImageView.dimensions();
	if (threadIdx.x < 16) {
		coord_t coord1 = coord_t(blockIdx.x * 16 + threadIdx.x, blockIdx.y * 16 + 15);
		coord_t coord2 = coord_t(blockIdx.x * 16 + threadIdx.x, blockIdx.y * 16 + 16);

		val1 = aImageView[coord1];
		val2 = aImageView[coord2];
	} else {
		coord_t coord1 = coord_t(blockIdx.x * 16 + 15, blockIdx.y * 16 + threadIdx.x - 16);
		coord_t coord2 = coord_t(blockIdx.x * 16 + 16, blockIdx.y * 16 + threadIdx.x - 16);

		val1 = aImageView[coord1];
		val2 = aImageView[coord2];
	}

	if (val1 && val2) {
		addEquivalence(aLut, min(val1, val2), max(val1, val2));

		/*value_type old = atomicMin(&aLut[val1], m);
		if (old < m) {
			atomicMin(&aLut[val1], m);
		}


		atomicMin(&aLut[val2], m);*/
	}

}

template <typename TImageView, typename TLUT>
void
merge_ccl_blocks(TImageView aImageView, TLUT aLut)
{
	dim3 blockSize(16, 16, 1);
	dim3 processingGridSize(
			(aImageView.dimensions().template get<0>() / blockSize.x + 1) - 1, 
			(aImageView.dimensions().template get<1>() / blockSize.y + 1) - 1, 
			1);
	dim3 processingBlockSize(32, 1, 1);


	//TODO Fix shared memory buffer size template parameter settings
	merge_ccl_blocks_kernel<TImageView, TLUT><<<processingGridSize, processingBlockSize>>>(aImageView, aLut);
}
//-----------------------------------------------------------------------------
}//namespace detail


template <typename TImageView, typename TLabelView, typename TLUT>
void 
connected_component_labeling(TImageView aImageView, TLabelView aLabelView, TLUT aLut)
{
	detail::init_labels(aImageView, aLabelView);

	detail::block_ccl(aLabelView);

	cudaMemset(&aLut[0], 0, aLut.dimensions() * sizeof(typename TLUT::value_type));
	detail::init_lut(aLabelView, aLut);

	detail::merge_ccl_blocks(aLabelView, aLut);

	detail::update_lut(aLut);

	detail::update_labels(aLabelView, aLut);
}


}//namespace cugip
