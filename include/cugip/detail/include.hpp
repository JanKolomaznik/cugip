#pragma once

// can be used in nvcc code and in normal code

#ifdef __CUDACC__
#	include <cuda.h>
#endif // __CUDACC__

#include <iostream>
#include <iomanip>
#include <boost/format.hpp>
#include <cassert>
#include <cstdio>


#ifndef DOUT
	#define DOUT	std::cout
#endif //DOUT


#ifndef NDEBUG
	#define D_PRINT( ARG )	\
		DOUT << "+++ " << ARG << std::endl; DOUT.flush();

#else
	#define D_PRINT( ARG )
#endif
