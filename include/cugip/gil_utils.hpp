#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/utils.hpp>

#include <boost/gil/image_view.hpp>

namespace cugip {

template <typename TLoc, typename TToView>
void copy(
	boost::gil::image_view<TLoc> from_view,
	TToView to_view)
{
	D_PRINT("*******************************");
	//copyAsync(from_view, to_view);
	//CUGIP_CHECK_RESULT(cudaThreadSynchronize());
}


}//namespace cugip
