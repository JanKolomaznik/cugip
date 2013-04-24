#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>

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
		lut[idx] = linear_access(aImageView, idx);// = outBuffer.mData[idx] != 0 ? idx+1 : 0;
	}
}

template <typename TImageView, typename TLUTBufferView>
void
init_lut(TImageView aImageView, TLUTBufferView aLUT)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	init_lut_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLUT);
}
//-----------------------------------------------------------------------------

template<typename TInputType, typename TLUT>
struct scan_neighborhood_ftor
{
	scan_neighborhood_ftor(TLUT aLUT, device_flag_view aLutUpdatedFlag) 
		: mLUT(aLUT), mLutUpdatedFlag(aLutUpdatedFlag)
	{}

	template<typename TLocator>
	CUGIP_DECL_HYBRID TOutputType
	minValidLabel(TLocator aLocator, TInputType aCurrent, const dimension_2d_tag &)
	{
		TInputType minimum = aCurrent;
		for (int j = -1; j < 1; ++j) {
			for (int i = -1; i < 1; ++i) {
				TInputType value = aLocator[typename TLocator::diff_t(i, j)];
				minimum = (value < minimum && value > 0) ? value : minimum;
			}
		}
		return minimum;
	}

	template<typename TLocator>
	CUGIP_DECL_HYBRID void
	operator()(TLocator aLocator) const
	{
		TInputType current = aLocator();
		if (0 < current) {
			TInputType minLabel = minValidLabel(aLocator, current, typename dimension<TLocator>::type());;
			mLUT[current-1] = minLabel < mLUT[current-1] ? minLabel : mLUT[current-1];

			mLutUpdatedFlag.set();
		}
	}

	TLUT mLUT;
	device_flag_view mLutUpdatedFlag;
};


template <typename TImageView, typename TLUTBufferView>
void
scan_image(TImageView aImageView, TLUTBufferView aLUT, device_flag_view aLutUpdatedFlag)
{
	filter(aImageView, scan_neighborhood_ftor(aLUT, aLutUpdatedFlag));
}
//-----------------------------------------------------------------------------
template <typename TImageView, typename TLUTBufferView>
void
update_lut_kernel(TImageView aImageView, TLUTBufferView aLUT)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;
	uint32 label, ref;

	if (idx < multiply(aImageView.dimensions())) {
		label = linear_access(aImageView, idx);

		if (label == idx+1) {		
			ref = label-1;
			label = lut[idx];
			while (ref != label-1) {
				ref = label-1;
				label = lut[ref];
			}
			lut[idx] = label;
		}
	}
}


template <typename TImageView, typename TLUTBufferView>
void
update_lut(TImageView aImageView, TLUTBufferView aLUT)
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (multiply(aImageView.dimensions()) + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	update_lut_kernel<<< gridSize1D, blockSize1D >>>(aImageView, aLUT);
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
			linear_access(aImageView, idx) = aLUT[label-1];
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

}//namespace detail


template <typename TImageView, typename TLUTBufferView>
void 
connected_component_labeling(TImageView aImageView, TLUTBufferView aLUT)
{
	//int lutUpdated = 0;

	device_flag lutUpdatedFlag;

	lutUpdatedFlag.reset();
        //cudaMemcpyToSymbol( "lutUpdated", &(lutUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

	detail::init_lut(aImageView, aLUT);
	detail::scan_image(aImageView, aLUT);

	//cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
	while (/*lutUpdated != 0*/lutUpdatedFlag) {
                //cudaMemcpyToSymbol( "lutUpdated", &(lutUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

		detail::update_lut(aImageView, aLUT);
		detail::update_labels(aImageView, aLUT);
		detail::scan_image(aImageView, aLUT, lutUpdatedFlag.view());
		
		//cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
	}
}

}//namespace cugip
