#pragma once
#include <type_traits>
#include <utility>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cugip/traits.hpp>

namespace cugip {

#ifndef REMOVE_PARENTHESES
	#define REMOVE_PARENTHESES(...) __VA_ARGS__
#endif

#define CUGIP_DECLARE_ARRAY_VIEW_TRAITS(CLASS, IS_DEVICE, IS_HOST, ...)\
	template<__VA_ARGS__>\
	struct is_array_view<REMOVE_PARENTHESES CLASS> : public std::true_type {};\
	template<__VA_ARGS__>\
	struct is_device_view<REMOVE_PARENTHESES CLASS> : public std::integral_constant<bool, IS_DEVICE> {};\
	template<__VA_ARGS__>\
	struct is_host_view<REMOVE_PARENTHESES CLASS> : public std::integral_constant<bool, IS_HOST> {};\
	template<__VA_ARGS__>\
	struct dimension<REMOVE_PARENTHESES CLASS>: dimension_helper<1> {};

#if defined(__CUDACC__)
template<typename TElement>
class DeviceArrayView;

template<typename TElement>
class DeviceArrayConstView {
public:
	static constexpr int cDimension = 1;
	using extents_t = int64_t;
	using coord_t = int64_t;
	using diff_t = int64_t;
	using value_type = typename std::remove_const<TElement>::type;
	using const_value_type = const value_type;
	using accessed_type = const value_type &;

	DeviceArrayConstView() = default;

	CUGIP_DECL_HYBRID
	DeviceArrayConstView(const_value_type *pointer, extents_t size) :
		mPointer(pointer),
		mSize(size)
	{}

	CUGIP_DECL_HYBRID
	DeviceArrayConstView(const DeviceArrayConstView<value_type> &view) :
		mPointer(view.pointer()),
		mSize(view.size())
	{}

	CUGIP_DECL_HYBRID
	DeviceArrayConstView(const DeviceArrayConstView<const_value_type> &view):
		mPointer(view.mPointer),
		mSize(view.mSize)
	{}

	CUGIP_DECL_HYBRID
	DeviceArrayConstView(const DeviceArrayView<const_value_type> &view);

	CUGIP_DECL_HYBRID
	DeviceArrayConstView<TElement> &operator=(const DeviceArrayConstView<value_type> &view) {
		mPointer = view.mPointer;
		mSize = view.mSize;
		return *this;
	}

	CUGIP_DECL_HYBRID
	DeviceArrayConstView<TElement> &operator=(const DeviceArrayConstView<const_value_type> &view) {
		mPointer = view.mPointer;
		mSize = view.mSize;
		return *this;
	}

	CUGIP_DECL_HYBRID
	DeviceArrayConstView<TElement> &operator=(const DeviceArrayView<value_type> &view);

	CUGIP_DECL_HYBRID
	extents_t size()  const {
		return mSize;
	}

	CUGIP_DECL_HYBRID
	extents_t dimensions()  const {
		return mSize;
	}

	CUGIP_DECL_HYBRID
	extents_t elementCount()  const {
		return mSize;
	}

	CUGIP_DECL_HYBRID
	bool empty() const {
		return mSize == 0;
	}

	CUGIP_DECL_HYBRID
	const_value_type *pointer() const {
		return mPointer;
	}

	CUGIP_DECL_DEVICE
	accessed_type operator[](coord_t index) const {
		CUGIP_ASSERT(index >= 0);
		CUGIP_ASSERT(index < mSize);
		return mPointer[index];
	}

	/// Creates view for part of this view.
	DeviceArrayConstView<TElement> subview(coord_t from, extents_t size) const {
		CUGIP_ASSERT(from >= 0);
		CUGIP_ASSERT(size <= mSize);
		CUGIP_ASSERT(from + size <= mSize);

		return DeviceArrayConstView(mPointer + from, size);
	}

	int strides() const {
		return 1;
	}

	CUGIP_DECL_HYBRID
	const_value_type *begin() const {
		return pointer();
	}


	CUGIP_DECL_HYBRID
	const_value_type *end() const {
		return pointer() + mSize;
	}


protected:
	const_value_type *mPointer = nullptr;
	extents_t mSize = 0;
};


template<typename TElement>
class DeviceArrayView: public DeviceArrayConstView<TElement> {
public:
	static constexpr int cDimension = 1;
	using extents_t = int64_t;
	using coord_t = int64_t;
	using diff_t = int64_t;
	using value_type = typename std::remove_const<TElement>::type;
	using const_value_type = const value_type;
	using accessed_type = value_type &;

	using predecessor_type = DeviceArrayConstView<TElement>;

	DeviceArrayView() = default;

	CUGIP_DECL_HYBRID
	DeviceArrayView(value_type *pointer, extents_t size) :
		predecessor_type(pointer, size)
	{}


	DeviceArrayView(const DeviceArrayView &) = default;

	DeviceArrayView &operator=(const DeviceArrayView &) = default;

	CUGIP_DECL_DEVICE
	accessed_type operator[](coord_t index) const {
		CUGIP_ASSERT(index >= 0);
		CUGIP_ASSERT(index < this->mSize);
		return const_cast<value_type *>(this->mPointer)[index];
	}

	CUGIP_DECL_HYBRID
	value_type *pointer() {
		return const_cast<value_type *>(this->mPointer);
	}

	CUGIP_DECL_HYBRID
	value_type *pointer() const {
		return this->mPointer;
	}

	/// Creates view for part of this view.
	DeviceArrayView<TElement> subview(coord_t from, extents_t size) const {
		CUGIP_ASSERT(from >= 0);
		CUGIP_ASSERT(size <= this->mSize);
		CUGIP_ASSERT(from + size <= this->mSize);

		return DeviceArrayView(const_cast<value_type *>(this->mPointer) + from, size);
	}

	DeviceArrayConstView<TElement> const_subview(coord_t from, extents_t size) const {
		return predecessor_type::subview(from, size);
	}

	CUGIP_DECL_HYBRID
	value_type *begin() {
		return const_cast<value_type *>(this->mPointer);
	}

	CUGIP_DECL_HYBRID
	value_type *end() {
		return const_cast<value_type *>(this->mPointer + this->mSize);
	}

};


template<typename TElement>
CUGIP_DECL_HYBRID
DeviceArrayConstView<TElement>::DeviceArrayConstView(const DeviceArrayView<const_value_type> &view):
	mPointer(view.pointer()),
	mSize(view.size())
{}

template<typename TElement>
CUGIP_DECL_HYBRID
DeviceArrayConstView<TElement> &
DeviceArrayConstView<TElement>::operator=(const DeviceArrayView<value_type> &view) {
	mPointer = view.pointer();
	mSize = view.size();
	return *this;
}


template<typename TElement>
auto view(thrust::device_vector<TElement> &buffer) {
	return DeviceArrayView<TElement>(buffer.data().get(), int64_t(buffer.size()));
}

template<typename TElement>
auto const_view(const thrust::device_vector<TElement> &buffer) {
	return DeviceArrayConstView<TElement>(buffer.data().get(), int64_t(buffer.size()));
}

/** \ingroup  traits
 * @{
 **/

CUGIP_DECLARE_ARRAY_VIEW_TRAITS((DeviceArrayConstView<TElement>), true, false, typename TElement)
CUGIP_DECLARE_ARRAY_VIEW_TRAITS((DeviceArrayView<TElement>), true, false, typename TElement)

/**
 * @}
 **/

#endif  // __CUDACC__

template<typename TElement>
class HostArrayView;

template<typename TElement>
class HostArrayConstView {
public:
	static constexpr int cDimension = 1;
	using extents_t = int64_t;
	using coord_t = int64_t;
	using diff_t = int64_t;
	using value_type = typename std::remove_const<TElement>::type;
	using const_value_type = const value_type;
	using accessed_type = const value_type &;

	HostArrayConstView() = default;

	HostArrayConstView(const_value_type *pointer, extents_t size) :
		mPointer(pointer),
		mSize(size)
	{}

	HostArrayConstView(const HostArrayConstView<value_type> &view) :
		mPointer(view.pointer()),
		mSize(view.size())
	{}

	HostArrayConstView(const HostArrayConstView<const_value_type> &view):
		mPointer(view.pointer()),
		mSize(view.size())
	{}

	HostArrayConstView(const HostArrayView<value_type> &view);

	HostArrayConstView<TElement> &operator=(const HostArrayConstView<value_type> &view) {
		mPointer = view.mPointer;
		mSize = view.mSize;
		return *this;
	}

	HostArrayConstView<TElement> &operator=(const HostArrayView<value_type> &view);

	HostArrayConstView<TElement> &operator=(const HostArrayConstView<const_value_type> &view) {
		mPointer = view.mPointer;
		mSize = view.mSize;
		return *this;
	}

	const_value_type *begin() const {
		return this->mPointer;
	}

	const_value_type *end() const {
		return this->mPointer + this->mSize;
	}


	extents_t size()  const {
		return mSize;
	}

	extents_t dimensions()  const {
		return mSize;
	}

	extents_t elementCount()  const {
		return mSize;
	}

	bool empty() const {
		return mSize == 0;
	}

	const_value_type *pointer() const {
		return mPointer;
	}

	accessed_type operator[](coord_t index) const {
		return mPointer[index];
	}

	/// Creates view for part of this view.
	HostArrayConstView<TElement> subview(coord_t from, extents_t size) const {
		CUGIP_ASSERT(from >= 0);
		CUGIP_ASSERT(size <= mSize);
		CUGIP_ASSERT(from + size <= mSize);

		return HostArrayConstView(mPointer + from, size);
	}

	int strides() const {
		return 1;
	}

protected:
	const_value_type *mPointer = nullptr;
	extents_t mSize = 0;
};


template<typename TElement>
class HostArrayView: public HostArrayConstView<TElement> {
public:
	static constexpr int cDimension = 1;
	using extents_t = int64_t;
	using coord_t = int64_t;
	using diff_t = int64_t;
	using value_type = typename std::remove_const<TElement>::type;
	using const_value_type = const value_type;
	using accessed_type = value_type &;

	using predecessor_type = HostArrayConstView<TElement>;

	HostArrayView() = default;

	HostArrayView(TElement *pointer, extents_t size) :
		predecessor_type(pointer, size)
	{}

	HostArrayView(const HostArrayView &) = default;
	HostArrayView(HostArrayView &&) = default;
	~HostArrayView() = default;

	HostArrayView &operator=(const HostArrayView &) = default;
	HostArrayView &operator=(HostArrayView &&) = default;


	accessed_type operator[](coord_t index) const {
		return const_cast<value_type *>(this->mPointer)[index];
	}

	value_type *pointer() {
		return const_cast<value_type *>(this->mPointer);
	}

	const_value_type *pointer() const {
		return this->mPointer;
	}

	value_type *begin() {
		return const_cast<value_type *>(this->mPointer);
	}

	value_type *end() {
		return const_cast<value_type *>(this->mPointer + this->mSize);
	}

	/// Creates view for part of this view.
	HostArrayView<TElement> subview(coord_t from, extents_t size) const {
		CUGIP_ASSERT(from >= 0);
		CUGIP_ASSERT(size <= this->mSize);
		CUGIP_ASSERT(from + size <= this->mSize);

		return HostArrayView(const_cast<value_type *>(this->mPointer) + from, size);
	}

	HostArrayConstView<TElement> const_subview(coord_t from, extents_t size) const {
		return predecessor_type::subview(from, size);
	}

};

template<typename TElement>
HostArrayConstView<TElement>::HostArrayConstView(const HostArrayView<value_type> &view):
	mPointer(view.pointer()),
	mSize(view.size())
{}


template<typename TElement>
HostArrayConstView<TElement> &
HostArrayConstView<TElement>::operator=(const HostArrayView<value_type> &view) {
	mPointer = view.mPointer;
	mSize = view.mSize;
	return *this;
}


template<typename TElement>
HostArrayView<TElement>
view(std::vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<const TElement>
const_view(std::vector<TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
const_view(std::vector<const TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayView<TElement>
view(thrust::host_vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
const_view(thrust::host_vector<const TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
const_view(thrust::host_vector<TElement> &buffer) {
	return HostArrayConstView<TElement>(buffer.data(), int64_t(buffer.size()));
}

/** \ingroup  traits
 * @{
 **/

CUGIP_DECLARE_ARRAY_VIEW_TRAITS((HostArrayConstView<TElement>), false, true, typename TElement)
CUGIP_DECLARE_ARRAY_VIEW_TRAITS((HostArrayView<TElement>), false, true, typename TElement)

/**
 * @}
 **/


//TODO - size instead of last
CUGIP_HD_WARNING_DISABLE
template<typename TView>
CUGIP_DECL_HYBRID
TView arraySubview(TView view, int64_t first, int64_t last) {
	auto pointer = view.pointer() + first;
	return TView(pointer, last - first);
}

template<typename TElement>
HostArrayConstView<TElement>
subview(HostArrayConstView<TElement> view, int64_t first, int64_t last) {
	return arraySubview(view, first, last);
}

template<typename TElement>
HostArrayView<TElement>
subview(HostArrayView<TElement> view, int64_t first, int64_t last) {
	return arraySubview(view, first, last);
}

#if defined(__CUDACC__)

template<typename TElement>
CUGIP_DECL_HYBRID
DeviceArrayConstView<TElement>
subview(DeviceArrayConstView<TElement> view, int64_t first, int64_t last) {
	return arraySubview(view, first, last);
}

template<typename TElement>
CUGIP_DECL_HYBRID
DeviceArrayView<TElement>
subview(DeviceArrayView<TElement> view, int64_t first, int64_t last) {
	return arraySubview(view, first, last);
}
#endif  // __CUDACC__

}  // namespace cugip

