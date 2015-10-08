#pragma once

#include <cugip/detail/include.hpp>
#include <cassert>

#include <map>

#include <boost/exception/all.hpp>
#include <boost/filesystem.hpp>

namespace detail {

/// ends recursion
inline void formatHelper(boost::format &aFormat) {}

/// Static recursion for format filling
template <typename T, typename... TArgs>
void formatHelper(boost::format &aFormat, T &&aValue, TArgs &&...aArgs) {
	aFormat % aValue;
	formatHelper(aFormat, std::forward<TArgs>(aArgs)...);
}

}  // detail

namespace cugip {


typedef boost::error_info<struct tag_message, std::string> MessageErrorInfo;

/// Error info containing file path.
typedef boost::error_info<struct tag_filename, boost::filesystem::path> FilenameErrorInfo;

class ExceptionBase: public virtual boost::exception, public virtual std::exception {};

class IncompatibleViewSizes: public ExceptionBase {};

#define CUGIP_DPRINT(...)\
	do { \
		std::cout << __VA_ARGS__ << std::endl; \
	} while (false);


/**
 * Logging with boost::format syntax.
 * CUGIP_DFORMAT("Format string arg1 = %1%; arg2 = %2%", 1, "two");
 **/
#define CUGIP_DFORMAT(format_string, ...) \
	do { \
		boost::format format(format_string); \
		::detail::formatHelper(format, ##__VA_ARGS__); \
		std::cout << __FILE__ << ":" << __LINE__ << ":"; \
		std::cout << format << std::endl; \
		std::cout.flush(); \
	} while (0)


#define CUGIP_THROW(...)\
	throw __VA_ARGS__;

// Error checking for cuda code
#ifdef __CUDACC__

	#define CUGIP_CHECK_RESULT_MSG( aErrorMessage, ... ) \
	{\
		cudaError_t err = __VA_ARGS__ ;\
		if( cudaSuccess != err ) {\
			std::string msg = boost::str(boost::format("%1%:%2%: %3% (%4%) %5%") % __FILE__ % __LINE__ % aErrorMessage % cugip::get_error_enum_name(err) % cudaGetErrorString(err));\
			D_PRINT( msg ); \
			CUGIP_THROW(std::runtime_error(msg));\
		}\
	}

	#define CUGIP_CHECK_RESULT( ... ) \
		CUGIP_CHECK_RESULT_MSG( #__VA_ARGS__, __VA_ARGS__ )

	#define CUGIP_CHECK_ERROR_STATE( aErrorMessage ) \
		CUGIP_CHECK_RESULT_MSG( aErrorMessage, cudaGetLastError() );

	#include <cugip/detail/error_codes.hpp>

#endif //__CUDACC__

}//namespace cugip
