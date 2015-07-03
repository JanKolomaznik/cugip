#pragma once

#include <cugip/detail/include.hpp>
#include <cassert>

#include <map>

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


#define CUGIP_STORE_ENUM_IN_MAP(MAP, ENUM)\
	MAP[ENUM] = #ENUM;

#define CUGIP_STRINGIFY_ENUM(ENUM)\
	std::cout << "(" << ENUM << "): " << #ENUM <<"\n"

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

inline std::map<cudaError_t, std::string>
generate_error_enum_names()
{
	std::map<cudaError_t, std::string> enumMap;
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaSuccess);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorMissingConfiguration);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorMemoryAllocation);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInitializationError);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorLaunchFailure);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorPriorLaunchFailure);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorLaunchTimeout);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorLaunchOutOfResources);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidDeviceFunction);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidConfiguration);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidDevice);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidValue);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidPitchValue);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidSymbol);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorMapBufferObjectFailed);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorUnmapBufferObjectFailed);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidHostPointer);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidDevicePointer);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidTexture);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidTextureBinding);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidChannelDescriptor);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidMemcpyDirection);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorAddressOfConstant);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorTextureFetchFailed);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorTextureNotBound);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorSynchronizationError);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidFilterSetting);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidNormSetting);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorMixedDeviceExecution);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorCudartUnloading);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorUnknown);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorNotYetImplemented);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorMemoryValueTooLarge);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidResourceHandle);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorNotReady);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInsufficientDriver);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorSetOnActiveProcess);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidSurface);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorNoDevice);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorECCUncorrectable);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorSharedObjectSymbolNotFound);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorSharedObjectInitFailed);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorUnsupportedLimit);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorDuplicateVariableName);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorDuplicateTextureName);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorDuplicateSurfaceName);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorDevicesUnavailable);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorInvalidKernelImage);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorNoKernelImageForDevice);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorIncompatibleDriverContext);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorPeerAccessAlreadyEnabled);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorPeerAccessNotEnabled);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorDeviceAlreadyInUse);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorProfilerDisabled);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorProfilerNotInitialized);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorProfilerAlreadyStarted);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorProfilerAlreadyStopped);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorAssert);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorTooManyPeers);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorHostMemoryAlreadyRegistered);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorHostMemoryNotRegistered);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorOperatingSystem);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorStartupFailure);
	CUGIP_STORE_ENUM_IN_MAP(enumMap, cudaErrorApiFailureBase);
	return enumMap;
}

inline std::string
get_error_enum_name(cudaError_t aError)
{
	static const std::map<cudaError_t, std::string> cEnumMap = generate_error_enum_names();
	std::map<cudaError_t, std::string>::const_iterator it = cEnumMap.find(aError);
	if (it != cEnumMap.end()) {
		return it->second;
	}
	return std::string();
}

inline void
print_error_enums() {
	CUGIP_STRINGIFY_ENUM(cudaSuccess);
	CUGIP_STRINGIFY_ENUM(cudaErrorMissingConfiguration);
	CUGIP_STRINGIFY_ENUM(cudaErrorMemoryAllocation);
	CUGIP_STRINGIFY_ENUM(cudaErrorInitializationError);
	CUGIP_STRINGIFY_ENUM(cudaErrorLaunchFailure);
	CUGIP_STRINGIFY_ENUM(cudaErrorPriorLaunchFailure);
	CUGIP_STRINGIFY_ENUM(cudaErrorLaunchTimeout);
	CUGIP_STRINGIFY_ENUM(cudaErrorLaunchOutOfResources);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidDeviceFunction);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidConfiguration);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidDevice);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidValue);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidPitchValue);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidSymbol);
	CUGIP_STRINGIFY_ENUM(cudaErrorMapBufferObjectFailed);
	CUGIP_STRINGIFY_ENUM(cudaErrorUnmapBufferObjectFailed);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidHostPointer);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidDevicePointer);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidTexture);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidTextureBinding);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidChannelDescriptor);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidMemcpyDirection);
	CUGIP_STRINGIFY_ENUM(cudaErrorAddressOfConstant);
	CUGIP_STRINGIFY_ENUM(cudaErrorTextureFetchFailed);
	CUGIP_STRINGIFY_ENUM(cudaErrorTextureNotBound);
	CUGIP_STRINGIFY_ENUM(cudaErrorSynchronizationError);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidFilterSetting);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidNormSetting);
	CUGIP_STRINGIFY_ENUM(cudaErrorMixedDeviceExecution);
	CUGIP_STRINGIFY_ENUM(cudaErrorCudartUnloading);
	CUGIP_STRINGIFY_ENUM(cudaErrorUnknown);
	CUGIP_STRINGIFY_ENUM(cudaErrorNotYetImplemented);
	CUGIP_STRINGIFY_ENUM(cudaErrorMemoryValueTooLarge);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidResourceHandle);
	CUGIP_STRINGIFY_ENUM(cudaErrorNotReady);
	CUGIP_STRINGIFY_ENUM(cudaErrorInsufficientDriver);
	CUGIP_STRINGIFY_ENUM(cudaErrorSetOnActiveProcess);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidSurface);
	CUGIP_STRINGIFY_ENUM(cudaErrorNoDevice);
	CUGIP_STRINGIFY_ENUM(cudaErrorECCUncorrectable);
	CUGIP_STRINGIFY_ENUM(cudaErrorSharedObjectSymbolNotFound);
	CUGIP_STRINGIFY_ENUM(cudaErrorSharedObjectInitFailed);
	CUGIP_STRINGIFY_ENUM(cudaErrorUnsupportedLimit);
	CUGIP_STRINGIFY_ENUM(cudaErrorDuplicateVariableName);
	CUGIP_STRINGIFY_ENUM(cudaErrorDuplicateTextureName);
	CUGIP_STRINGIFY_ENUM(cudaErrorDuplicateSurfaceName);
	CUGIP_STRINGIFY_ENUM(cudaErrorDevicesUnavailable);
	CUGIP_STRINGIFY_ENUM(cudaErrorInvalidKernelImage);
	CUGIP_STRINGIFY_ENUM(cudaErrorNoKernelImageForDevice);
	CUGIP_STRINGIFY_ENUM(cudaErrorIncompatibleDriverContext);
	CUGIP_STRINGIFY_ENUM(cudaErrorPeerAccessAlreadyEnabled);
	CUGIP_STRINGIFY_ENUM(cudaErrorPeerAccessNotEnabled);
	CUGIP_STRINGIFY_ENUM(cudaErrorDeviceAlreadyInUse);
	CUGIP_STRINGIFY_ENUM(cudaErrorProfilerDisabled);
	CUGIP_STRINGIFY_ENUM(cudaErrorProfilerNotInitialized);
	CUGIP_STRINGIFY_ENUM(cudaErrorProfilerAlreadyStarted);
	CUGIP_STRINGIFY_ENUM(cudaErrorProfilerAlreadyStopped);
	CUGIP_STRINGIFY_ENUM(cudaErrorAssert);
	CUGIP_STRINGIFY_ENUM(cudaErrorTooManyPeers);
	CUGIP_STRINGIFY_ENUM(cudaErrorHostMemoryAlreadyRegistered);
	CUGIP_STRINGIFY_ENUM(cudaErrorHostMemoryNotRegistered);
	CUGIP_STRINGIFY_ENUM(cudaErrorOperatingSystem);
	CUGIP_STRINGIFY_ENUM(cudaErrorStartupFailure);
	CUGIP_STRINGIFY_ENUM(cudaErrorApiFailureBase);
}


#define CUGIP_THROW(...)\
	throw __VA_ARGS__;

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


}//namespace cugip

