#pragma once

#include <cugip/detail/include.hpp>

namespace cugip {

#define CUGIP_THROW(...)\
	throw __VA_ARGS__;

#define CUGIP_CHECK_RESULT_MSG( aErrorMessage, ... ) \
{\
	cudaError_t err = __VA_ARGS__ ;\
	if( cudaSuccess != err ) {\
		std::string msg = boost::str(boost::format("%1%:(%2%) %3%") % aErrorMessage % err % cudaGetErrorString(err));\
		D_PRINT( msg ); \
		CUGIP_THROW(std::runtime_error(msg));\
	}\
}

#define CUGIP_CHECK_RESULT( ... ) \
	CUGIP_CHECK_RESULT_MSG( #__VA_ARGS__, __VA_ARGS__ )

#define CUGIP_CHECK_ERROR_STATE( aErrorMessage ) \
	CUGIP_CHECK_RESULT_MSG( aErrorMessage, cudaGetLastError() );

//TODO - provide assertion
//#define CUGIP_ASSERT_RESULT( ... ) \
//	CUGIP_CHECK_RESULT_MSG( "Assertion failure!", __VA_ARGS__ )

#define CUGIP_STRINGIFY_ENUM(ENUM)\
	std::cout << "(" << ENUM << "): " << #ENUM <<"\n"

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

}//namespace cugip

