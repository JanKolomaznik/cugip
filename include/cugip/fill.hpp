#pragma once

#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>

namespace cugip {


template <typename TView, typename TType>
void
fill(TView aView, TType aValue)
{
	cugip::copy_async(constantImage(aValue, aView.dimensions()), aView);
}

}//namespace cugip

