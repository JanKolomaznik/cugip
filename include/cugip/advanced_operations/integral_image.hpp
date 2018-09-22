#pragma once

#include <cugip/scan.hpp>

namespace cugip {


template<typename TInputView, typename TOutputView>
void integral_image(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, /*TODO execution policy*/)
{
	scan(aInput, aTmpView);
	scan(aTmpView, aOutput);
}


} // namespace cugip
