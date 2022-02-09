#pragma once

// Guard to allow logger definitions to be included
#define CUGIP_LOG_INCLUDED
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

#ifdef CUGIP_USE_BOOST_LOG

	#include <cugip/detail/boost_logging.hpp>

#else //CUGIP_USE_BOOST_LOG

	#include <cugip/detail/default_logging.hpp>

#endif //CUGIP_USE_BOOST_LOG


#ifndef NDEBUG
#define CUGIP_DPRINT(...)\
	do { \
		CUGIP_DEBUG_LOGGER << __VA_ARGS__ CUGIP_LOG_NEWLINE; \
	} while (false);

#else
#define CUGIP_DPRINT(...)
#endif //NDEBUG

#define CUGIP_TPRINT(...)\
	do { \
		CUGIP_TRACE_LOGGER << __VA_ARGS__ CUGIP_LOG_NEWLINE; \
	} while (false);

#define CUGIP_IPRINT(...)\
	do { \
		CUGIP_INFO_LOGGER << __VA_ARGS__ CUGIP_LOG_NEWLINE; \
	} while (false);

#define CUGIP_WPRINT(...)\
	do { \
		CUGIP_WARNING_LOGGER << __VA_ARGS__ CUGIP_LOG_NEWLINE; \
	} while (false);

#define CUGIP_EPRINT(...)\
	do { \
		CUGIP_ERROR_LOGGER << __VA_ARGS__ CUGIP_LOG_NEWLINE; \
	} while (false);



#define CUGIP_LOG_FORMAT(logger, format_string, ...) \
	do { \
		boost::format format(format_string); \
		::detail::formatHelper(format, ##__VA_ARGS__); \
		logger << __FILE__ << ":" << __LINE__ << ":" \
			<< format CUGIP_LOG_NEWLINE; \
	} while (0)

#ifndef NDEBUG
/**
 * Logging with boost::format syntax.
 * CUGIP_DFORMAT("Format string arg1 = %1%; arg2 = %2%", 1, "two");
 **/
#define CUGIP_DFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_DEBUG_LOGGER, format_string, __VA_ARGS__)

#else
#define CUGIP_DFORMAT(...)
#endif //NDEBUG


#define CUGIP_TFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_TRACE_LOGGER, format_string, __VA_ARGS__)

#define CUGIP_IFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_INFO_LOGGER, format_string, __VA_ARGS__)

#define CUGIP_WFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_WARNING_LOGGER, format_string, __VA_ARGS__)

#define CUGIP_EFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_ERROR_LOGGER, format_string, __VA_ARGS__)
