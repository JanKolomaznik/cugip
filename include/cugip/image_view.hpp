#pragma once

#include <cugip/detail/include.hpp>
#include <boost/mpl/bool.hpp>

namespace cugip {

template<typename TElement, size_t tDim = 2>
class device_image_view
{
public:

protected:

};

template<typename TElement, size_t tDim = 2>
class const_device_image_view
{
public:

protected:

};

template<typename TView>
struct is_device_view: public boost::mpl::false_
{
	/*typedef boost::mpl::false_ type;
	static const bool value = type::value;*/
};

template<typename TElement, size_t tDim>
struct is_device_view<device_image_view<TElement, tDim> > : public boost::mpl::true_
{
	/*typedef boost::mpl::true_ type;
	static const bool value = type::value;*/
};

}//namespace cugip
