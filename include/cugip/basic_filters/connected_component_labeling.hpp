#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/access_utils.hpp>

#include <cugip/neighborhood.hpp>


namespace cugip {

namespace detail {

//-----------------------------------------------------------------------------
template <typename TImageView, typename TLUTBufferView>
CUGIP_GLOBAL void
init_lut_kernel(TImageView aImageView, TLUTBufferView aLUT)
{
	int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if (idx < multiply(aImageView.dimensions())) {
		//TODO - check 0
		aLUT.set(idx + 1, linear_access(aImageView, idx));// = outBuffer.mData[idx] != 0 ? idx+1 : 0;
	}
}

template <typename TImageView, typename TLUTBufferView>
void
init_lut(TImageView aImageView, TLUTBufferView aLut)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	init_lut_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLUT);
}

template<typename TInputType, typename TLUT>
struct scan_neighborhood_for_connections_ftor
{
	scan_neighborhood_for_connections_ftor(TLUT aLUT, device_flag_view aLutUpdatedFlag)
		: mLUT(aLUT), mLutUpdatedFlag(aLutUpdatedFlag)
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID TInputType
	minValidLabel(TLocator aLocator, TInputType aCurrent, const dimension_2d_tag &)
	{
		typedef full_neighborhood<dimension<TLocator>::value> neighborhood;
		TInputType minimum = aCurrent;

		for (int i = 0; i < neighborhood::count; ++i) {
			TInputType value = aLocator[neighborhood::get(i)/*typename TLocator::diff_t(i, j)*/];
			minimum = (value < minimum && value > 0) ? value : minimum;
		}

		/*for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				TInputType value = aLocator[typename TLocator::diff_t(i, j)];
				minimum = (value < minimum && value > 0) ? value : minimum;
			}
		}*/
		return minimum;
	}

	template<typename TLocator>
	CUGIP_DECL_DEVICE void // TODO hybrid
	operator()(TLocator aLocator)
	{
		TInputType current = aLocator.get();
		if (0 < current) {
			TInputType minLabel = minValidLabel(aLocator, current, typename dimension<TLocator>::type());;
			if (minLabel < mLUT.get(current)) {
				//printf("%d - %d - %d; ", current, mLUT[current-1], minLabel);
				mLUT.set(current, minLabel);
				mLutUpdatedFlag.set_device();
			}
		}
	}
}

template <typename TLUTBufferView>
void
scan_image_for_connections(TImageView aImageView, TLUTBufferView aLUT, device_flag_view aLutUpdatedFlag)
{
	for_each_locator(aImageView, scan_neighborhood_for_connections_ftor<typename TImageView::value_type, TLUTBufferView>(aLUT, aLutUpdatedFlag));
}

//-----------------------------------------------------------------------------
template <typename TImageView, typename TLUTBufferView>
CUGIP_GLOBAL void
update_lut_kernel(TImageView aImageView, TLUTBufferView aLUT)
{
	int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if (idx < multiply(aImageView.dimensions())) {
		int label = linear_access(aImageView, idx);

		if (label == idx+1) {
			aLUT.compress(label);
			/*int ref = label-1;
			label = aLUT[idx];
			while (ref != label-1) {
				ref = label-1;
				label = aLUT[ref];
			}
			aLUT[idx] = label;*/
		}
	}
}

template <typename TImageView, typename TLUTBufferView>
void
update_labels(TImageView aImageView, TLUTBufferView aLUT)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	update_lut_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLUT);
}
//-----------------------------------------------------------------------------
template <typename TImageView, typename TLabelView>
CUGIP_GLOBAL void
init_labels_kernel(TImageView aImageView, TLabelView aLabelView)
{
	int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if (idx < multiply(aImageView.dimensions())) {
		uint64_t label = linear_access(aImageView, idx);
		if ( label > 0 ) {
			linear_access(aImageView, idx) = aLUT.get(label);
		}
	}
}

template <typename TImageView, typename TLabelView>
void
init_labels(TImageView aImageView, TLabelView aLabelView)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	update_labels_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLUT);
}


//-----------------------------------------------------------------------------
template <typename TImageView, size_t tSharedMemoryBufferSize>
CUGIP_GLOBAL void
block_ccl_kernel(TImageView aImageView)
{
	typedef typename TImageView::value_type value_type;
	typedef typename TImageView::coord_t coord_t;
	typedef typename TImageView::extents_t extents_t;
	typedef device_image_view<value_type, dimension<TImageView>::value> tmp_image_t;
	//TODO 3D version
	coord_t coord = coord_t(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	extents_t extents = aImageView.dimensions();

	CUGIP_ASSERT(tSharedMemoryBufferSize >= (blockDim.x * blockDim.y * blockDim.z));

	coord_t threadCoord(threadIdx.x, threadIdx.y, threadIdx.z);
	extents_t blockExtents(blockDim.x, blockDim.y, blockDim.z);
	CUGIP_SHARED value_type blockData[tSharedMemoryBufferSize];
	tmp_image_t blockView = cugip::view(device_ptr<value_type>(blockData), blockExtents);

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
		typedef typename get_neighborhood_accessor<tmp_image_t, neighborhood_4_tag>::type neighborhood;
		neighborhood acc = neighborhood(blockView.template locator<border_handling_repeat_t>(threadCoord));
		changed = 0;
		for (size_t i = 0; i < neighborhood::size; ++i) {
			if (current != 0) {
				value_type newValue = acc[i];
				if ((newValue < current) && (newValue != 0)) {
					blockView[threadCoord] = current =  newValue;
					++changed;
				}
			}
			__syncthreads();
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


template <typename TImageView, typename TLUTBufferView>
void
connected_component_labeling(TImageView aImageView, TLUTBufferView aLUT)
{
	device_flag lutUpdatedFlag;
	lutUpdatedFlag.reset_host();

	D_PRINT("CCL initialization ...");
	detail::init_lut(aImageView, aLUT);
	detail::scan_image_for_connections(aImageView, aLUT, lutUpdatedFlag.view());
/*detail::update_lut(aImageView, aLUT);
		detail::update_labels(aImageView, aLUT);*/

	int i = 0;
	while (lutUpdatedFlag.check_host()) {
	//for (int i = 0; i < 1000; ++i) {
		D_PRINT("    Running CCL iteration ..." << ++i);
		lutUpdatedFlag.reset_host();

		detail::update_lut(aImageView, aLUT);
		detail::update_labels(aImageView, aLUT);
		detail::scan_image_for_connections(aImageView, aLUT, lutUpdatedFlag.view());
	}
	D_PRINT("CCL done!");
}

template<typename TOutputValue, int TDimension>
struct assign_masked_id_functor
{
	assign_masked_id_functor(typename dim_traits<TDimension>::extents_t aExtents)
		: extents(aExtents)
	{}

	template<typename TInputValue>
	CUGIP_DECL_HYBRID TOutputValue
	operator()(const TInputValue &aArg, typename dim_traits<TDimension>::coord_t aCoordinates)const
	{
		if (aArg == 0) {
			return 0;
		}
		return 1 + get_linear_access_index(extents, aCoordinates);
	}
	typename dim_traits<TDimension>::extents_t extents;
};


template <typename TInImageView, typename TOutImageView>
void
assign_masked_ids(TInImageView aInput, TOutImageView aOutput)
{
	cugip::transform_position(
			aInput,
			aOutput,
			cugip::assign_masked_id_functor<typename TOutImageView::value_type, dimension<TOutImageView>::value>(aInput.dimensions()));
}


}//namespace cugip
