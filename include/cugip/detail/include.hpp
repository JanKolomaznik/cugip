#pragma once

#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <boost/format.hpp>
#include <cassert>
#include <cstdio>
#include <boost/call_traits.hpp>
#include <boost/type_traits.hpp>


#ifndef DOUT
	#define DOUT	std::cout
#endif //DOUT


#ifndef NDEBUG
	#define D_PRINT( ARG )	\
		DOUT << "+++ " << ARG << std::endl; DOUT.flush();
#else
	#define D_PRINT( ARG )
#endif
