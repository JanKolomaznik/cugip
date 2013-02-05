#pragma once


template<typename TType>
struct negate
{
	CUGIL_DECL_HYBRID TType 
	operator()(const TType &aArg)
	{
		return -aArg;
	}
};
