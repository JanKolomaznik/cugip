#pragma once

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

#define CUGIP_DEBUG_LOGGER std::cout
#define CUGIP_TRACE_LOGGER std::cout
#define CUGIP_ERROR_LOGGER std::cerr
#define CUGIP_INFO_LOGGER std::cout



#define CUGIP_DPRINT(...)\
	do { \
		CUGIP_DEBUG_LOGGER << __VA_ARGS__ << std::endl; \
	} while (false);

#define CUGIP_TPRINT(...)\
	do { \
		CUGIP_TRACE_LOGGER << __VA_ARGS__ << std::endl; \
	} while (false);

#define CUGIP_EPRINT(...)\
	do { \
		CUGIP_TRACE_LOGGER << __VA_ARGS__ << std::endl; \
	} while (false);

#define CUGIP_PRINT(...)\
	do { \
		CUGIP_INFO_LOGGER << __VA_ARGS__ << std::endl; \
	} while (false);

#define CUGIP_LOG_FORMAT(logger, format_string, ...) \
	do { \
		boost::format format(format_string); \
		::detail::formatHelper(format, ##__VA_ARGS__); \
		logger << __FILE__ << ":" << __LINE__ << ":" \
			<< format << std::endl; \
	} while (0)

/**
 * Logging with boost::format syntax.
 * CUGIP_DFORMAT("Format string arg1 = %1%; arg2 = %2%", 1, "two");
 **/
#define CUGIP_DFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_DEBUG_LOGGER, format_string, __VA_ARGS__)

#define CUGIP_TFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_TRACE_LOGGER, format_string, __VA_ARGS__)

#define CUGIP_EFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_ERROR_LOGGER, format_string, __VA_ARGS__)

#define CUGIP_LFORMAT(format_string, ...) \
	CUGIP_LOG_FORMAT(CUGIP_INFO_LOGGER, format_string, __VA_ARGS__)
