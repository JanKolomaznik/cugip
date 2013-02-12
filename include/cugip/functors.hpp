#pragma once

#include <cugip/detail/include.hpp>

namespace cugip {

template<typename TType>
struct negate
{
	CUGIL_DECL_HYBRID TType 
	operator()(const TType &aArg)
	{
		return -aArg;
	}
};

}//namespace cugip
