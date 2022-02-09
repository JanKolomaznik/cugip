#pragma once

#ifndef CUGIP_LOG_INCLUDED
	#error This file cannot be included by itself - include through "detail/logging.hpp".
#endif

#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/trivial.hpp>

BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(cugip_logger, boost::log::sources::severity_logger_mt< >)

#define CUGIP_TRACE_LOGGER BOOST_LOG_SEV(cugip_logger::get(), ::boost::log::trivial::trace)
#define CUGIP_DEBUG_LOGGER BOOST_LOG_SEV(cugip_logger::get(), ::boost::log::trivial::debug)
#define CUGIP_INFO_LOGGER BOOST_LOG_SEV(cugip_logger::get(), ::boost::log::trivial::info)
#define CUGIP_ERROR_LOGGER BOOST_LOG_SEV(cugip_logger::get(), ::boost::log::trivial::error)
#define CUGIP_WARNING_LOGGER BOOST_LOG_SEV(cugip_logger::get(), ::boost::log::trivial::warning)

#define CUGIP_LOG_NEWLINE

