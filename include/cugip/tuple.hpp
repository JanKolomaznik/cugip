#pragma once

#include <cugip/utils.hpp>

namespace cugip {

// helpers
template <typename T>
struct id { using type = T; };

template <typename T>
using type_of = typename T::type;

template <size_t... N>
struct Sizes : id<Sizes<N...>> { };

// choose N-th element in list <T...>
template <size_t N, typename... T>
struct Choose;

template <size_t N, typename H, typename... T>
struct Choose<N, H, T...> : Choose <N-1, T...> { };

template <typename H, typename... T>
struct Choose<0, H, T...> : id <H> { };

template <size_t N, typename... T>
using choose = type_of<Choose <N, T...> >;

// given L>=0, generate sequence <0, ..., L-1>
template <size_t L, size_t I = 0, typename S = Sizes<> >
struct Range;

template <size_t L, size_t I, size_t... N>
struct Range<L, I, Sizes<N...>> : Range<L, I+1, Sizes<N..., I> > { };

template <size_t L, size_t... N>
struct Range<L, L, Sizes<N...>> : Sizes<N...> { };

template <size_t L>
using range = type_of <Range <L>>;

// single tuple element
template <size_t N, typename T>
class TupleElem
{
	T elem;
public:
	CUGIP_DECL_HYBRID
	T&       get()       { return elem; }

	CUGIP_DECL_HYBRID
	const T& get() const { return elem; }
};

// tuple implementation
template <typename N, typename... T>
class TupleImpl;

template <size_t... N, typename... T>
class TupleImpl<Sizes<N...>, T...> : TupleElem<N, T>...
{
	template <size_t M> using pick = choose<M, T...>;
	//template <size_t M> using elem = TupleElem<M, pick<M>>;
	template <size_t M> using elem = TupleElem<M, choose<M, T...>>;

public:
	template <size_t M>
	CUGIP_DECL_HYBRID
	pick<M>& get() { return elem<M>::get(); }

	template <size_t M>
	CUGIP_DECL_HYBRID
	const pick<M>& get() const { return elem<M>::get(); }
};

template <typename... T>
struct Tuple : TupleImpl<range<sizeof...(T)>, T...>
{
	CUGIP_DECL_HYBRID
	static constexpr std::size_t size() { return sizeof...(T); }
};

} // namespace cugip
