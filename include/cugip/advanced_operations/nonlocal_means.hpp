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

template <typename TInImageView, typename TOutImageView, int tPatchRadius, int tSearchRadius>
CUGIP_GLOBAL void
kernel_nonlocal_means(TImageView aIn, TOutImageView aOut)
{
	typename TImageView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TImageView::extents_t extents = aIn.dimensions();

	if (coord < extents) {
		/*aOperator(
			aView1.template locator<cugip::border_handling_repeat_t>(coord),
			aView2.template locator<cugip::border_handling_repeat_t>(coord)
			);*/
	}
}

} // namespace detail

template <typename TInImageView, typename TOutImageView, int tPatchRadius, int tSearchRadius>
void
nonlocal_means(TImageView aIn, TOutImageView aOut)
{
	D_PRINT("nonlocal_means ...");

	D_PRINT("nonlocal_means done!");
}

}//namespace cugip
