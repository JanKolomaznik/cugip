#pragma once
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/access_utils.hpp>


namespace cugip {

template<typename TLabel>
class device_union_find_view
{
public:
	CUGIP_DECL_DEVICE void
	merge(TLabel aLabel1, TLabel aLabel2)
	{

	}

	CUGIP_DECL_DEVICE void
	find(TLabel aLabel)
	{
		while (aLabel != mBuffer[aLabel]) {
			aLabel = mBuffer[aLabel];
		}
		return aLabel;
	}

	CUGIP_DECL_HOST void
	compress()
	{

	}

protected:
	device_memory_1d mBuffer;
}

template<typename TLabel>
class device_union_find
{
public:
/*	typedef device_image_view<TElement, tDim> view_t;
	typedef const_device_image_view<TElement, tDim> const_view_t;
	typedef TElement element_t;
	typedef TElement value_type;

	typedef typename dim_traits<tDim>::extents_t extents_t;

	friend view_t view<>(device_image<TElement, tDim> &);
	friend const_view_t const_view<>(device_image<TElement, tDim> &);*/
public:
	device_union_find()
	{}

	device_union_find(size_t aLabelCount)
		: mData(aLabelCount)
	{}

	CUGIP_DECL_HYBRID size_t
	label_count() const
	{
		return mData.dimensions();
	}

	CUGIP_DECL_HOST device_union_find_view<TLabel>
	view()
	{
		return device_union_find_view<TLabel>(mData);
	}
protected:
	device_union_find & operator=(const device_union_find &);
	device_union_find(const device_union_find &);

	typename memory_management<TLabel, 1>::device_memory_owner mData;
};


} // namespace cugip
