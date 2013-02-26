#pragma once

namespace cugip {

template <typename TArg1View, typename TArg21View>
void 
add(TArg1View aArg1View, TArg21View aArg2View)
{
	CUGIP_ASSERT(aArg1View.dimensions() == aArg2View.dimensions());
}

template <typename TArg1View, typename TArg21View>
void 
subtract(TArg1View aArg1View, TArg21View aArg2View)
{
	CUGIP_ASSERT(aArg1View.dimensions() == aArg2View.dimensions());
}

}//namespace cugip


