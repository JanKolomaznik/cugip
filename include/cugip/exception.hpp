#pragma once

namespace cugip {
	
#define CUGIL_THROW(...)\
	throw __VA_ARGS__;

#define CUGIL_CHECK_RESULT_MSG( aErrorMessage, ... ) \
{\
	cudaError_t err = __VA_ARGS__ ;\
	if( cudaSuccess != err ) {\
		D_PRINT( aErrorMessage ); \
		CUGIL_THROW(std::runtime_error(aErrorMessage));\
	}\
}

#define CUGIL_CHECK_RESULT( ... ) \
	CUGIL_CHECK_RESULT_MSG( #__VA_ARGS__, __VA_ARGS__ )

#define CUGIL_CHECK_ERROR_STATE( aErrorMessage ) \
	CUDA_CHECK_RESULT_MSG( aErrorMessage, cudaGetLastError() );

//TODO - provide assertion
#define CUGIL_ASSERT_RESULT( ... ) \
	CUGIL_CHECK_RESULT_MSG( "Assertion failure!", __VA_ARGS__ )
}//namespace cugip
