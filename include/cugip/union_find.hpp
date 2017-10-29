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
	device_union_find_view(const device_memory_1d<TLabel> &aBuffer)
		: mBuffer(aBuffer)
	{}

	CUGIP_DECL_DEVICE void
	merge(TLabel aLabel1, TLabel aLabel2)
	{

	}

	CUGIP_DECL_DEVICE TLabel
	find(TLabel aLabel) const
	{
		while (aLabel != mBuffer[aLabel]) {
			aLabel = mBuffer[aLabel];
		}
		return aLabel;
	}

	CUGIP_DECL_DEVICE TLabel
	compress(TLabel aLabel)
	{
		TLabel label = find(aLabel);
		set(aLabel, label);
		return label;
	}


	CUGIP_DECL_DEVICE TLabel
	get(TLabel aLabel) const
	{
		return mBuffer[aLabel];
	}

	CUGIP_DECL_DEVICE void
	set(TLabel aLabel, TLabel aNewLabel)
	{
		mBuffer[aLabel] = aNewLabel;
	}

	/*CUGIP_DECL_HOST void
	compress()
	{

	}*/

protected:
	device_memory_1d<TLabel> mBuffer;
};

template<typename TLabel>
class device_union_find
{
public:
	typedef device_union_find_view<TLabel> view_t;
	//typedef const_device_image_view<TElement, tDim> const_view_t;
	typedef TLabel label_t;
public:
	device_union_find()
	{}

	device_union_find(int aLabelCount)
		: mData(aLabelCount + 1)
	{}

	CUGIP_DECL_HYBRID int
	label_count() const
	{
		return mData.dimensions() - 1;
	}

	CUGIP_DECL_HOST device_union_find_view<TLabel>
	view() const
	{
		return device_union_find_view<TLabel>(mData);
	}
protected:
	device_union_find & operator=(const device_union_find &);
	device_union_find(const device_union_find &);

	device_memory_1d_owner<TLabel> mData;
};
/*
template <typename TLabel>
device_union_find_view<TLabel>
view(const device_union_find<TLabel> &aUnionFind)
{
	return aUnionFind.view();
}*/


} // namespace cugip
