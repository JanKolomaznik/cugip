#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/detail/logging.hpp>
#include <cassert>

#include <map>

#include <boost/exception/all.hpp>
#include <boost/filesystem.hpp>

namespace cugip {


typedef boost::error_info<struct tag_message, std::string> MessageErrorInfo;

/// Error info containing file path.
typedef boost::error_info<struct tag_filename, boost::filesystem::path> FilenameErrorInfo;

class ExceptionBase: public virtual boost::exception, public virtual std::exception {};

class IncompatibleViewSizes: public ExceptionBase {};


#define CUGIP_THROW(...)\
	throw __VA_ARGS__;

// Error checking for cuda code
#ifdef __CUDACC__

	typedef boost::error_info<struct tag_cuda_error_code, cudaError_t> CudaErrorCodeInfo;

	#define CUGIP_CHECK_RESULT_MSG( aErrorMessage, ... ) \
	do {\
		cudaError_t err = __VA_ARGS__ ;\
		if( cudaSuccess != err ) {\
			CUGIP_EFORMAT("%1% (%2%) %3%", aErrorMessage, cugip::get_error_enum_name(err), cudaGetErrorString(err)); \
			CUGIP_THROW(ExceptionBase() << MessageErrorInfo(aErrorMessage) << CudaErrorCodeInfo(err));\
		}\
	} while(0);

	#define CUGIP_CHECK_RESULT( ... ) \
		CUGIP_CHECK_RESULT_MSG( #__VA_ARGS__, __VA_ARGS__ )

	#define CUGIP_CHECK_ERROR_STATE( aErrorMessage ) \
		CUGIP_CHECK_RESULT_MSG( aErrorMessage, cudaGetLastError() );

	#include <cugip/detail/error_codes.hpp>

#endif //__CUDACC__

}//namespace cugip
