#pragma once

#ifndef CUGIP_LOG_INCLUDED
	#error This file cannot be included by itself - include through "detail/logging.hpp".
#endif

#define CUGIP_DEBUG_LOGGER std::cout
#define CUGIP_TRACE_LOGGER std::cout
#define CUGIP_ERROR_LOGGER std::cerr
#define CUGIP_WARNING_LOGGER std::cerr
#define CUGIP_INFO_LOGGER std::cout


#define CUGIP_LOG_NEWLINE << "\n"
