#pragma once

#include <cugip/utils.hpp>

namespace cugip {

// helpers
template <typename T>
struct Id { using type = T; };

template <typename T>
using type_of = typename T::type;

template <size_t... N>
struct Sizes : Id<Sizes<N...>> { };

// choose N-th element in list <T...>
template <size_t N, typename... T>
struct Choose;

template <size_t N, typename H, typename... T>
struct Choose<N, H, T...> : Choose<N-1, T...> { };

template <typename H, typename... T>
struct Choose<0, H, T...> : Id<H> { };

template <size_t N, typename... T>
using choose = type_of<Choose<N, T...>>;

// given L>=0, generate sequence <0, ..., L-1>
template <size_t L, size_t I = 0, typename S = Sizes<>>
struct Range;

template <size_t L, size_t I, size_t... N>
struct Range<L, I, Sizes<N...>> : Range<L, I+1, Sizes<N..., I>> { };

template <size_t L, size_t... N>
struct Range<L, L, Sizes<N...>> : Sizes<N...> { };

template <size_t L>
using range = type_of<Range <L>>;

// single tuple element
CUGIP_HD_WARNING_DISABLE
template <size_t N, typename T>
class TupleElem
{
	T elem;
public:
	CUGIP_DECL_HYBRID constexpr
	TupleElem() {}

	CUGIP_DECL_HYBRID constexpr
	TupleElem(T aElement)
		: elem(aElement)
	{}

	CUGIP_DECL_HYBRID constexpr
	T& get() { return elem; }

	CUGIP_DECL_HYBRID constexpr
	const T& get() const { return elem; }
};

// tuple implementation
template <typename N, typename... T>
class TupleImpl;

CUGIP_HD_WARNING_DISABLE
template <size_t... N, typename... T>
class TupleImpl<Sizes<N...>, T...> : TupleElem<N, T>...
{
	template <size_t M> using pick = choose<M, T...>;
	//template <size_t M> using elem = TupleElem<M, pick<M>>;
	template <size_t M> using elem = TupleElem<M, choose<M, T...>>;

public:
	CUGIP_DECL_HYBRID constexpr
	TupleImpl() {}

	/*CUGIP_HD_WARNING_DISABLE*/ CUGIP_DECL_HYBRID constexpr
	TupleImpl(T... aItems)
		: TupleElem<N, T>(aItems)...
	{}

	template <size_t M>
	CUGIP_DECL_HYBRID constexpr
	pick<M>& get() { return elem<M>::get(); }

	template <size_t M>
	CUGIP_DECL_HYBRID constexpr
	const pick<M>& get() const { return elem<M>::get(); }
};

CUGIP_HD_WARNING_DISABLE
template <typename... T>
struct Tuple : TupleImpl<range<sizeof...(T)>, T...>
{
	typedef TupleImpl<range<sizeof...(T)>, T...> Predecessor;

	CUGIP_DECL_HYBRID constexpr
	Tuple(){}

	CUGIP_DECL_HYBRID constexpr
	Tuple(T... aItems)
		: Predecessor(aItems...)
	{}

	CUGIP_DECL_HYBRID
	static constexpr std::size_t size() { return sizeof...(T); }
};

template<int tIdx, typename... TType>
struct get_policy<tIdx, const Tuple<TType...>>
{
	typedef const typename Choose<tIdx, TType...>::type & return_type;
	typedef const Tuple<TType...> & value_t;

	static CUGIP_DECL_HYBRID constexpr auto
	get(value_t aArg) -> decltype(aArg.template get<tIdx>())
	{
		return aArg.template get<tIdx>();
	}
};

template<int tIdx, typename... TType>
struct get_policy<tIdx, Tuple<TType...>>
{
	typedef typename Choose<tIdx, TType...>::type & return_type;
	typedef Tuple<TType...> & value_t;

	static CUGIP_DECL_HYBRID constexpr auto
	get(value_t aArg) -> decltype(aArg.template get<tIdx>())
	{
		return aArg.template get<tIdx>();
	}
};

} // namespace cugip
