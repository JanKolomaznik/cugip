#pragma once

#include <cugip/exception.hpp>
#include <cugip/exception_error_info.hpp>
#include <cugip/detail/logging.hpp>

namespace cugip {

// Error checking for cuda code
#ifdef __CUDACC__

	typedef boost::error_info<struct tag_cuda_error_code, cudaError_t> CudaErrorCodeInfo;

	#define CUGIP_CHECK_RESULT_MSG( aErrorMessage, ... ) \
	do {\
		cudaError_t err = __VA_ARGS__ ;\
		if( cudaSuccess != err ) {\
			CUGIP_EFORMAT("%1% (%2%) %3%", aErrorMessage, cugip::get_error_enum_name(err), cudaGetErrorString(err)); \
			CUGIP_THROW(cugip::ExceptionBase() << cugip::MessageErrorInfo(aErrorMessage) << cugip::CudaErrorCodeInfo(err));\
		}\
	} while(0);

	#define CUGIP_CHECK_RESULT( ... ) \
		CUGIP_CHECK_RESULT_MSG( #__VA_ARGS__, __VA_ARGS__ )

	#define CUGIP_CHECK_ERROR_STATE( aErrorMessage ) \
		CUGIP_CHECK_RESULT_MSG( aErrorMessage, cudaGetLastError() );

	#include <cugip/detail/error_codes.hpp>

#endif //__CUDACC__


} // namespace cugip
