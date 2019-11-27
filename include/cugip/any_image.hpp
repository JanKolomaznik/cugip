#pragma once

#include <boost/variant.hpp>
#include <cugip/color_spaces.hpp>
#include <cugip/host_image.hpp>
#include <cugip/host_image_view.hpp>


namespace cugip {

using AnyImage = boost::variant<
			host_image<RGB_8, 2>
			>;

template<typename TVariant>
struct AnyImageViewTraits;

template<typename ...TImage>
struct AnyImageViewTraits<boost::variant<TImage...>> {
	using views = boost::variant<decltype(view(std::declval<TImage>()))...>;
	using const_views = boost::variant<decltype(const_view(std::declval<TImage>()))...>;
};

template<typename TResult>
struct AnyImageViewVisitor: public boost::static_visitor<TResult>
{
	template<typename TImage>
	TResult operator()(const TImage &aImage) const
	{
		return view(aImage);
	}
};

template<typename TResult>
struct AnyImageConstViewVisitor: public boost::static_visitor<TResult>
{
	template<typename TImage>
	TResult operator()(const TImage &aImage) const
	{
		return const_view(aImage);
	}
};

template<typename ...TImage>
auto view(const boost::variant<TImage...> &aImage)
{
	using Result = typename AnyImageViewTraits<boost::variant<TImage...>>::views;
	return boost::apply_visitor(AnyImageViewVisitor<Result>{}, aImage);
}

template<typename ...TImage>
auto const_view(const boost::variant<TImage...> &aImage)
{
	using Result = typename AnyImageViewTraits<boost::variant<TImage...>>::const_views;
	return boost::apply_visitor(AnyImageConstViewVisitor<Result>{}, aImage);
}

} // namespace cugip

