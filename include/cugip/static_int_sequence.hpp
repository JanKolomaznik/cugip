#pragma once

#include <initializer_list>
#include <cugip/math.hpp>

namespace cugip {

template<typename ...TArgs>
void ignoreReturnValues(TArgs ...) {}

template<int... tInts >
class IntSequence {
	constexpr int Size() const {
		return sizeof...(tInts);
	}
};


namespace detail {

template<int tValue, int tHead, int... tTail>
struct IsValueInIntSequenceImpl {
	static constexpr bool value = (tValue == tHead) || IsValueInIntSequenceImpl<tValue, tTail...>::value;
};

template<int tValue, int tHead>
struct IsValueInIntSequenceImpl<tValue, tHead> {
	static constexpr bool value = tValue == tHead;
};

template<int tCurrentIndex, int tValue, int tHead, int... tTail>
struct GetPositionInIntSequenceImpl {
	static constexpr int value = tValue == tHead ? tCurrentIndex : GetPositionInIntSequenceImpl<tCurrentIndex + 1, tValue, tTail...>::value;
};

template<int tCurrentIndex, int tValue, int tHead>
struct GetPositionInIntSequenceImpl<tCurrentIndex, tValue, tHead> {
	static constexpr int value = tValue == tHead ? tCurrentIndex : -1;
};

template<int tFront, int...tValues>
struct GenerateIntSequenceImpl {
	typedef typename GenerateIntSequenceImpl<tFront-1, tFront, tValues...>::Type Type;
};

template<int...tValues>
struct GenerateIntSequenceImpl<0, tValues...> {
	typedef IntSequence<0, tValues...> Type;
};

template<int tCurrentResult, typename TIntSequence>
struct ProductIntSequenceImpl;

template<int tCurrentResult, int tHead, int...tValues>
struct ProductIntSequenceImpl<tCurrentResult, IntSequence<tHead, tValues...>> {
	static constexpr int value = ProductIntSequenceImpl<tCurrentResult * tHead, IntSequence<tValues...>>::value;
};

template<int tCurrentResult, int tHead>
struct ProductIntSequenceImpl<tCurrentResult, IntSequence<tHead>> {
	static constexpr int value = tCurrentResult * tHead;
};

template<typename TIntSequence>
struct LastItemInIntSequenceImpl;

template<int tHead, int... tTail>
struct LastItemInIntSequenceImpl<IntSequence<tHead, tTail...>> {
	static constexpr int value = LastItemInIntSequenceImpl<IntSequence<tTail...>>::value;
};

template<int tTail>
struct LastItemInIntSequenceImpl<IntSequence<tTail>> {
	static constexpr int value = tTail;
};

}  // namespace detail


template<int tValue, typename TIntSequence>
struct IsValueInIntSequence {
	template<bool tDummy, typename TDummy>
	struct Helper;

	template<bool tDummy, int... tSequence>
	struct Helper<tDummy, IntSequence<tSequence...>> {
		static constexpr bool value = detail::IsValueInIntSequenceImpl<tValue, tSequence...>::value;
	};


	static constexpr bool value = Helper<false, TIntSequence>::value;
};


template<int tValue, typename TIntSequence>
struct GetPositionInIntSequence {
	static_assert(IsValueInIntSequence<tValue, TIntSequence>::value, "Value is not in the IntSequence!");

	template<bool tDummy, typename TDummy>
	struct Helper;

	template<bool tDummy, int... tSequence>
	struct Helper<tDummy, IntSequence<tSequence...>> {
		static constexpr int value = detail::GetPositionInIntSequenceImpl<0, tValue, tSequence...>::value;
	};


	static constexpr int value = Helper<false, TIntSequence>::value;
};

template<typename TIntSequence>
struct ProductOfIntSequence
{
	static constexpr int value = detail::ProductIntSequenceImpl<1, TIntSequence>::value;
};

template<typename TIntSequence>
struct LastItemInIntSequence
{
	static constexpr int value = detail::LastItemInIntSequenceImpl<TIntSequence>::value;
};

template<int tSize>
struct GenerateIntSequence {
	typedef typename detail::GenerateIntSequenceImpl<tSize - 1>::Type Type;
};

template<int tSize>
using MakeIntSequence = typename GenerateIntSequence<tSize>::Type;

template<int...tValues>
std::ostream &operator<<(std::ostream &stream, const IntSequence<tValues...> &) {
	stream << "[";

	(void) std::initializer_list<int>{((stream << tValues << ", "), 0)...};

	return stream << "]";
}

template<int...tSize>
struct StaticSize: IntSequence<tSize...>
{
	static constexpr int cDimension = sizeof...(tSize);

	/*CUGIP_DECL_HYBRID
	static constexpr simple_vector<int, cDimension>
	vector()
	{
		return simple_vector<int, cDimension>{ tSize... };
	}*/

	CUGIP_DECL_HYBRID
	static constexpr int count()
	{
		return ProductOfIntSequence<IntSequence<tSize...>>::value;
	}

	CUGIP_DECL_HYBRID
	static constexpr int last()
	{
		return LastItemInIntSequence<IntSequence<tSize...>>::value;
	}
};

// TODO make constexpr
template<int...tSize>
CUGIP_DECL_HYBRID
simple_vector<int, sizeof...(tSize)>
to_vector(IntSequence<tSize...>) {
	return simple_vector<int, sizeof...(tSize)>(tSize... );
	//return simple_vector<int, sizeof...(tSize)>{ tSize... };
}

template<int tDimension, int tValue>
struct FillStaticSize;

template<int tValue>
struct FillStaticSize<2, tValue>
{
	typedef StaticSize<tValue, tValue> Type;
};

template<int tValue>
struct FillStaticSize<3, tValue>
{
	typedef StaticSize<tValue, tValue, tValue> Type;
};

}  // namespace cugip
