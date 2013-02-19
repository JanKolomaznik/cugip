#pragma once

#include <cugip/detail/include.hpp>

namespace cugip {

template<typename TType>
struct negate
{
	CUGIL_DECL_HYBRID TType 
	operator()(const TType &aArg)const
	{
		TType tmp;
		tmp.data[0] = 0;//255 - aArg.data[0];
		tmp.data[1] = 0;//255 - aArg.data[1];
		tmp.data[2] = 255 - aArg.data[2];
		printf("1");
		return tmp;
	}
};

}//namespace cugip
