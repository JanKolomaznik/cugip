#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>
#include <cugip/image_locator.hpp>
#include <cugip/math.hpp>


#include <cugip/reduce.tcc>

namespace cugip {

struct ScanExecutionConfig {
	cudaStream_t stream = 0;
};


template<typename TInputView, typename TOutputView, typename TOutputValue, typename TOperator, int tDirection, typename TExecutionConfig>
void scan(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOutputValue aInitialValue, IntValue<tDirection> aDirection, TOperator aOperator, TExecutionConfig aExecutionConfig);

template<typename TInputView, typename TOutputView, typename TOutputValue, typename TOperator, int tDirection>
void scan(TInputView aInput, TOutputView aOutput, TOutputView aTmpView, TOutputValue aInitialValue, IntValue<tDirection> aDirection, TOperator aOperator);

} // namespace cugip



#include <cugip/scan.tcc>
