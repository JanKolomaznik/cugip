#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/detail/logging.hpp>
#include <cassert>

#include <map>

#include <boost/exception/all.hpp>

namespace cugip {



class ExceptionBase: public virtual boost::exception, public virtual std::exception
{
public:
	// const char* what() const noexcept override {
	// 	return boost::diagnostic_information(*this).c_str();
	// }
};

class EIncompatibleViewSizes: public ExceptionBase {};
class ESliceOutOfRange: public ExceptionBase {};
class EInvalidRange: public ExceptionBase {};


#define CUGIP_THROW(ex) BOOST_THROW_EXCEPTION(ex)


}//namespace cugip
