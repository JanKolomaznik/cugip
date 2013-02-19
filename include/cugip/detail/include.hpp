#pragma once

#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <boost/format.hpp>
#include <cassert>


#ifndef DOUT
	#define DOUT	std::cout
#endif //DOUT


#ifndef NDEBUG
	#define D_PRINT( ARG )	\
		DOUT << "+++ " << ARG << std::endl; DOUT.flush();
#else
	#define D_PRINT( ARG )
#endif
