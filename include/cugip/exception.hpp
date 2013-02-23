#pragma once

#include <cugip/detail/include.hpp>

namespace cugip {
	
#define CUGIP_THROW(...)\
	throw __VA_ARGS__;

#define CUGIP_CHECK_RESULT_MSG( aErrorMessage, ... ) \
{\
	cudaError_t err = __VA_ARGS__ ;\
	if( cudaSuccess != err ) {\
		std::string msg = boost::str(boost::format("%1%: %2%") % aErrorMessage % cudaGetErrorString(err));\
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


}//namespace cugip

