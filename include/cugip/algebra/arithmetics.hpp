#pragma once

namespace cugip {

namespace detail {
	
template<typename TWeight>
struct weighted_add_ftor
{
	weighted_add_ftor(TWeight aWeight): mWeight(aWeight)
	{}

	template <typename TArg1, typename TArg2>
	CUGIP_DECL_HYBRID void
	operator()(TArg1 &aArg1, const TArg2 &aArg2)
	{
		aArg1 += mWeight * aArg2;
	}
	TWeight mWeight;
};

}//namespace detail

template <typename TArg1View, typename TArg21View, typename TWeight>
void 
add(TArg1View aArg1View, TWeight aWeight, TArg21View aArg2View)
{
	CUGIP_ASSERT(aArg1View.dimensions() == aArg2View.dimensions());

	for_each(aArg1View, aArg2View, detail::weighted_add_ftor<TWeight>(aWeight));
}

template <typename TArg1View, typename TArg21View>
void 
subtract(TArg1View aArg1View, TArg21View aArg2View)
{
	CUGIP_ASSERT(aArg1View.dimensions() == aArg2View.dimensions());
}


namespace detail {
	
template<typename TType>
struct multiply_ftor
{
	multiply_ftor(TType aFactor): mFactor(aFactor)
	{}

	template <typename TInput>
	CUGIP_DECL_HYBRID TInput 
	operator()(const TInput &aArg)const
	{
		return mFactor * aArg;
	}

	TType mFactor;

};

}//namespace detail

template <typename TView, typename TType>
void multiply(TView aView, TType aFactor)
{
	for_each(aView, detail::multiply_ftor<TType>(aFactor));
}

}//namespace cugip


