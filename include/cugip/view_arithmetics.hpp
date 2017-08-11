#pragma once

#include <cugip/math.hpp>
#include <cugip/functors.hpp>
#include <cugip/tuple.hpp>
#include <cugip/detail/view_declaration_utils.hpp>

#include <cugip/image_locator.hpp>

namespace cugip {

/// Base class for image views implementing lazy evaluation of operators working on two image views.
template<typename TView1, typename TView2>
class BinaryOperatorDeviceImageView : public device_image_view_base<dimension<TView1>::value>
{
public:
	static_assert(dimension<TView1>::value == dimension<TView1>::value, "Both views must have same dimension!");
	typedef device_image_view_base<dimension<TView1>::value> predecessor_type;

	typedef typename TView1::value_type value1_type;
	typedef typename TView2::value_type value2_type;

	BinaryOperatorDeviceImageView(TView1 view1, TView2 view2) :
		predecessor_type(view1.dimensions()),
		mView1(view1),
		mView2(view2)
	{
		/* TODO if (view1.Size() != view2.Size()) {
			THROW
		}*/
	}

protected:
	TView1 mView1;
	TView2 mView2;
};


/// Image view, which returns linear combination of elements from two other image views.
/// R = f1 * I1 + f2 * I2
template<typename TFactor1, typename TView1, typename TFactor2, typename TView2>
class LinearCombinationImageView : public BinaryOperatorDeviceImageView<TView1, TView2> {
public:
	typedef BinaryOperatorDeviceImageView<TView1, TView2> predecessor_type;

	typedef typename TView1::value_type value1_type;
	typedef typename TView2::value_type value2_type;
	typedef typename std::common_type<TFactor1, value1_type>::type result1_type;
	typedef typename std::common_type<TFactor2, value2_type>::type result2_type;
	typedef typename std::common_type<result1_type, result2_type>::type result_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(result_type, dimension<TView1>::value)

	LinearCombinationImageView(TFactor1 factor1, TView1 view1, TFactor2 factor2, TView2 view2) :
		predecessor_type(view1, view2),
		mFactor1(factor1),
		mFactor2(factor2)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	value_type operator[](coord_t index) const {
		return mFactor1 * this->mView1[index] + mFactor2 * this->mView2[index];
	}

protected:
	TFactor1 mFactor1;
	TFactor2 mFactor2;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(LinearCombinationImageView<TFactor1, TView1, TFactor2, TView2>),
	dimension<TView1>::value,
	(is_device_view<TView1>::value && is_device_view<TView2>::value),
	(is_host_view<TView1>::value && is_host_view<TView2>::value),
	typename TFactor1, typename TView1, typename TFactor2, typename TView2);

/// Utility function to create LinearCombinationImageView without the need to specify template parameters.
template<typename TFactor1, typename TView1, typename TFactor2, typename TView2>
LinearCombinationImageView<TFactor1, TView1, TFactor2, TView2>
linearCombination(TFactor1 factor1, TView1 view1, TFactor2 factor2, TView2 view2) {
	return LinearCombinationImageView<TFactor1, TView1, TFactor2, TView2>(factor1, view1, factor2, view2);
}


/// Utility function wrapping two image addition.
template<typename TView1, typename TView2>
LinearCombinationImageView<int, TView1, int, TView2>
add(TView1 view1, TView2 view2) {
	// TODO(johny) - possible more efficient implementation
	return LinearCombinationImageView<int, TView1, int, TView2>(1, view1, 1, view2);
}


/// Utility function wrapping two image subtraction.
template<typename TView1, typename TView2>
LinearCombinationImageView<int, TView1, int, TView2>
subtract(TView1 view1, TView2 view2) {
	// TODO(johny) - possible more efficient implementation
	return LinearCombinationImageView<int, TView1, int, TView2>(1, view1, -1, view2);
}


/// Image view, which returns per element multiplication.
/// R = I1 .* I2
template<typename TView1, typename TView2>
class MultiplicationImageView : public BinaryOperatorDeviceImageView<TView1, TView2> {
public:
	typedef BinaryOperatorDeviceImageView<TView1, TView2> predecessor_type;
	typedef typename TView1::value_type value_type1;
	typedef typename TView2::value_type value_type2;
	typedef decltype(std::declval<value_type1>() * std::declval<value_type2>()) result_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(result_type, dimension<TView1>::value)

	MultiplicationImageView(TView1 view1, TView2 view2) :
		predecessor_type(view1, view2)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return this->mView1[index] * this->mView2[index];
	}
};

CUGIP_DECLARE_VIEW_TRAITS(
	(MultiplicationImageView<TView1, TView2>),
	dimension<TView1>::value,
	(is_device_view<TView1>::value && is_device_view<TView2>::value),
	(is_host_view<TView1>::value && is_host_view<TView2>::value),
	typename TView1, typename TView2);

/// Utility function to create MultiplicationImageView without the need to specify template parameters.
template<typename TView1, typename TView2>
MultiplicationImageView<TView1, TView2>
multiply(TView1 view1, TView2 view2) {
	return MultiplicationImageView<TView1, TView2>(view1, view2);
}

template<typename TView1, typename TView2>
class MaskedImageView : public BinaryOperatorDeviceImageView<TView1, TView2> {
public:
	typedef BinaryOperatorDeviceImageView<TView1, TView2> predecessor_type;
	typedef typename TView1::value_type value_type1;
	typedef typename TView2::value_type value_type2;
	typedef value_type1 result_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(result_type, dimension<TView1>::value)

	MaskedImageView(TView1 view1, TView2 view2, value_type1 aDefaultValue) :
		predecessor_type(view1, view2),
		mDefaultValue(aDefaultValue)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		if (this->mView2[index]) {
			return this->mView1[index];
		}
		return mDefaultValue;
	}

	value_type1 mDefaultValue;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(MaskedImageView<TView1, TView2>),
	dimension<TView1>::value,
	(is_device_view<TView1>::value && is_device_view<TView2>::value),
	(is_host_view<TView1>::value && is_host_view<TView2>::value),
	typename TView1, typename TView2);

/// Utility function to create MultiplicationImageView without the need to specify template parameters.
template<typename TView1, typename TView2>
MaskedImageView<TView1, TView2>
maskView(TView1 view1, TView2 view2, typename TView1::value_type aDefaultValue) {
	return MaskedImageView<TView1, TView2>(view1, view2, aDefaultValue);
}


/// Image view, which returns per element division.
/// R = I1 ./ I2
template<typename TView1, typename TView2>
class DivisionImageView : public BinaryOperatorDeviceImageView<TView1, TView2> {
public:
	typedef BinaryOperatorDeviceImageView<TView1, TView2> predecessor_type;
	typedef typename TView1::value_type value_type1;
	typedef typename TView2::value_type value_type2;
	typedef decltype(std::declval<value_type1>() / std::declval<value_type2>()) result_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(result_type, dimension<TView1>::value)

	typedef typename TView1::Element Element1;
	typedef typename TView2::Element Element2;


	typedef typename std::common_type<Element1, Element2>::type Element;
	typedef Element AccessType;

	DivisionImageView(TView1 view1, TView2 view2) :
		predecessor_type(view1, view2)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	Element operator[](coord_t index) const {
		return this->mView1[index] / this->mView2[index];
	}
};

CUGIP_DECLARE_VIEW_TRAITS(
	(DivisionImageView<TView1, TView2>),
	dimension<TView1>::value,
	(is_device_view<TView1>::value && is_device_view<TView2>::value),
	(is_host_view<TView1>::value && is_host_view<TView2>::value),
	typename TView1, typename TView2);


/// Utility function to create DivisionImageView without the need to specify template parameters.
template<typename TView1, typename TView2>
DivisionImageView<TView1, TView2>
divide(TView1 view1, TView2 view2) {
	return DivisionImageView<TView1, TView2>(view1, view2);
}


/// View which allows mirror access to another view
/// TODO(johny) - specialization for memory based views - only stride and pointer reordering
template<typename TView>
class MirrorImageView : public device_image_view_base<dimension<TView>::value> {
public:
	typedef device_image_view_base<dimension<TView>::value> predecessor_type;
	// TODO - if possible reference access
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	MirrorImageView(TView view, simple_vector<bool, cDimension> flips) :
		predecessor_type(view.dimensions()),
		view_(view),
		flips_(flips)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return view_[flipCoordinates(index, this->dimensions(), flips_)];
	}

protected:
	CUGIP_DECL_HYBRID
	static coord_t flipCoordinates(coord_t index, extents_t size, simple_vector<bool, cDimension> flips) {
		for (int i = 0; i < cDimension; ++i) {
			if (flips[i]) {
				index[i] = size[i] - index[i] - 1;
			}
		}
		return index;
	}

	TView view_;
	simple_vector<bool, cDimension> flips_;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(MirrorImageView<TView>),
	dimension<TView>::value,
	(is_device_view<TView>::value),
	(is_host_view<TView>::value),
	typename TView);



/// Create mirror views with fliped axes specification
template<typename TView>
MirrorImageView<TView>
mirror(TView view, simple_vector<bool, TView::kDimension> flips) {
	return MirrorImageView<TView>(view, flips);
}


/// View which pads another image view
/// TODO(johny) - do also tapper padding
template<typename TView/*, bool tIsPeriodic*/>
class PaddedImageView : public device_image_view_base<dimension<TView>::value> {
public:
	typedef device_image_view_base<dimension<TView>::value> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(typename TView::value_type, dimension<TView>::value)

	PaddedImageView(TView view, const extents_t &size, const extents_t &offset, value_type fill_value) :
		predecessor_type(size),
		view_(view),
		offset_(offset),
		fill_value_(fill_value)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		index = ModPeriodic(index - offset_, this->dimensions());
		if (view_.IsIndexInside(index)) {
			return view_[index];
		}
		return fill_value_;
	}

protected:
	TView view_;
	simple_vector<int, cDimension> offset_;
	value_type fill_value_;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(PaddedImageView<TView>),
	dimension<TView>::value,
	(is_device_view<TView>::value),
	(is_host_view<TView>::value),
	typename TView);


/// Create padded view
template<typename TView>
PaddedImageView<TView>
padView(
	TView view,
	simple_vector<int, dimension<TView>::value> size,
	simple_vector<int, dimension<TView>::value> offset,
	typename TView::value_type fill_value)
{
	return PaddedImageView<TView>(view, size, offset, fill_value);
}


/// View which allows mirror access to another view
/// TODO(johny) - specialization for memory based views - only stride and pointer reordering
template<typename TView, typename TOperator>
class UnaryOperatorImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		UnaryOperatorImageView<TView, TOperator>>
{
public:
	typedef UnaryOperatorImageView<TView, TOperator> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(std::declval<TOperator>()(std::declval<Input>())) result_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(result_type, dimension<TView>::value)

	UnaryOperatorImageView(TView view, TOperator unary_operator) :
		predecessor_type(view.dimensions()),
		mView(view),
		mUnaryOperator(unary_operator)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return mUnaryOperator(mView[index]);
	}

protected:
	TView mView;
	TOperator mUnaryOperator;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(UnaryOperatorImageView<TView, TOperator>),
	dimension<TView>::value,
	(is_device_view<TView>::value),
	(is_host_view<TView>::value),
	typename TView, typename TOperator);

/// Creates view which returns squared values from the original view
template<typename TView>
UnaryOperatorImageView<TView, SquareFunctor>
square(TView view) {
	return UnaryOperatorImageView<TView, SquareFunctor>(view, SquareFunctor());
}

template<typename TView>
UnaryOperatorImageView<TView, AbsFunctor>
abs_view(TView view) {
	return UnaryOperatorImageView<TView, AbsFunctor>(view, AbsFunctor());
}

/// Creates view which returns square root of the values from the original view
template<typename TView>
UnaryOperatorImageView<TView, SquareRootFunctor>
squareRoot(TView view) {
	return UnaryOperatorImageView<TView, SquareRootFunctor>(view, SquareRootFunctor());
}


/// Utility function to create multiplied view without the need to specify template parameters.
template<typename TFactor, typename TView>
UnaryOperatorImageView<TView, MultiplyByFactorFunctor<TFactor>>
multiplyByFactor(TFactor factor, TView view) {
	return UnaryOperatorImageView<TView, MultiplyByFactorFunctor<TFactor>>(view, MultiplyByFactorFunctor<TFactor>(factor));
}

/// Creates view returning values from the original view with value added.
template<typename TType, typename TView>
UnaryOperatorImageView<TView, AddValueFunctor<TType>>
addValue(TType value, TView view) {
	return UnaryOperatorImageView<TView, AddValueFunctor<TType>>(view, AddValueFunctor<TType>(value));
}


/// Returns view returning values from the original view with values lower then limit replaced by the limit
template<typename TType, typename TView>
UnaryOperatorImageView<TView, LowerLimitFunctor<TType>>
lowerLimit(TType limit, TView view) {
	return UnaryOperatorImageView<TView, LowerLimitFunctor<TType>>(view, LowerLimitFunctor<TType>(limit));
}


/// Returns view returning values from the original view with values lower then limit replaced by the specified replacement
template<typename TType, typename TView>
UnaryOperatorImageView<TView, LowerLimitReplacementFunctor<TType>>
lowerLimit(TType limit, TType replacement, TView view) {
	return UnaryOperatorImageView<TView, LowerLimitReplacementFunctor<TType>>(view, LowerLimitReplacementFunctor<TType>(limit, replacement));
}


/// Returns view returning values from the original view with values bigger then limit replaced by the limit
template<typename TType, typename TView>
UnaryOperatorImageView<TView, UpperLimitFunctor<TType>>
upperLimit(TType limit, TView view) {
	return UnaryOperatorImageView<TView, UpperLimitFunctor<TType>>(view, UpperLimitFunctor<TType>(limit));
}


/// Returns view returning values from the original view with values bigger then limit replaced by the specified replacement
template<typename TType, typename TView>
UnaryOperatorImageView<TView, UpperLimitReplacementFunctor<TType>>
upperLimit(TType limit, TType replacement, TView view) {
	return UnaryOperatorImageView<TView, UpperLimitReplacementFunctor<TType>>(view, UpperLimitReplacementFunctor<TType>(limit, replacement));
}

template<typename TFunctor, typename TView>
UnaryOperatorImageView<TView, TFunctor>
unaryOperator(TView view, TFunctor functor) {
	return UnaryOperatorImageView<TView, TFunctor>(view, functor);
}

template<typename TView, typename TOperator>
class UnaryOperatorOnPositionImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		UnaryOperatorOnPositionImageView<TView, TOperator>>
{
public:
	typedef UnaryOperatorOnPositionImageView<TView, TOperator> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(std::declval<TOperator>()(std::declval<Input>(), typename TView::coord_t())) result_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(result_type, dimension<TView>::value)

	UnaryOperatorOnPositionImageView(TView view, TOperator unary_operator) :
		predecessor_type(view.dimensions()),
		mView(view),
		mUnaryOperator(unary_operator)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return mUnaryOperator(mView[index], index);
	}

protected:
	TView mView;
	TOperator mUnaryOperator;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(UnaryOperatorOnPositionImageView<TView, TOperator>),
	dimension<TView>::value,
	(is_device_view<TView>::value),
	(is_host_view<TView>::value),
	typename TView, typename TOperator);


template<typename TFunctor, typename TView>
UnaryOperatorOnPositionImageView<TView, TFunctor>
unaryOperatorOnPosition(TView view, TFunctor functor) {
	return UnaryOperatorOnPositionImageView<TView, TFunctor>(view, functor);
}

template<typename TView, typename TOperator, typename TBorderHandling = cugip::BorderHandlingTraits<border_handling_enum::REPEAT>>
class UnaryOperatorOnLocatorImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		UnaryOperatorOnLocatorImageView<TView, TOperator, TBorderHandling>>
{
public:
	typedef UnaryOperatorOnLocatorImageView<TView, TOperator, TBorderHandling> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(std::declval<TOperator>()(std::declval<image_locator<TView, TBorderHandling>>())) result_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(result_type, dimension<TView>::value)

	//TODO - support border handling
	UnaryOperatorOnLocatorImageView(TView view, TOperator unary_operator) :
		predecessor_type(view.dimensions()),
		mView(view),
		mUnaryOperator(unary_operator)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return mUnaryOperator(mView.template locator<TBorderHandling>(index));
	}

protected:
	TView mView;
	TOperator mUnaryOperator;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(UnaryOperatorOnLocatorImageView<TView, TOperator, TBorderHandling>),
	dimension<TView>::value,
	(is_device_view<TView>::value),
	(is_host_view<TView>::value),
	typename TView, typename TOperator, typename TBorderHandling);



template<typename TFunctor, typename TView>
UnaryOperatorOnLocatorImageView<TView, TFunctor>
unaryOperatorOnLocator(TView view, TFunctor functor) {
	return UnaryOperatorOnLocatorImageView<TView, TFunctor>(view, functor);
}

template<typename TFirstView, typename... TViews>
struct MultiViewTraits
{
	static const int cDimension = dimension<TFirstView>::value;

	typedef typename TFirstView::extents_t extents_t;
	typedef typename TFirstView::coord_t coord_t;
	typedef typename TFirstView::diff_t diff_t;
	static const bool cIsDeviceView = fold_and<is_device_view, TFirstView, TViews...>::value;
	static const bool cIsHostView = fold_and<is_host_view, TFirstView, TViews...>::value;
};

/*CUGIP_HD_WARNING_DISABLE
template<typename TCoordinates, typename TOperator, typename TTuple, size_t ...N>
CUGIP_DECL_HYBRID
int call(TCoordinates aIndex, const TOperator &aOperator, const TTuple &aViews, Sizes<N...> aIndices)
{
	return aOperator(aViews.get<N>()[aIndex]...);
}*/


template<typename TOperator, typename TView, typename... TViews>
class NAryOperatorDeviceImageView
	: public device_image_view_crtp<
		MultiViewTraits<TView, TViews...>::cDimension,
		NAryOperatorDeviceImageView<TOperator, TView, TViews...>>
{
public:
	typedef MultiViewTraits<TView, TViews...> MultiTraits;

	typedef typename MultiTraits::extents_t extents_t;
	typedef typename MultiTraits::coord_t coord_t;
	typedef typename MultiTraits::diff_t diff_t;
	typedef NAryOperatorDeviceImageView<TOperator, TView, TViews...> this_t;
	typedef device_image_view_crtp<
		MultiViewTraits<TView, TViews...>::cDimension,
		NAryOperatorDeviceImageView<TOperator, TView, TViews...>> predecessor_type;
	//typedef typename TView::value_type Input;
	typedef decltype(std::declval<TOperator>()(std::declval<typename TView::value_type>(), std::declval<typename TViews::value_type>()...)) result_type;
	typedef result_type value_type;
	typedef const result_type const_value_type;
	typedef result_type accessed_type;

	NAryOperatorDeviceImageView(TOperator aOperator, TView aView, TViews... aViews) :
		predecessor_type(aView.dimensions()),
		mOperator(aOperator),
		mViews(aView, aViews...)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const
	{
		return call(index/*, mOperator, mViews*/, typename Range<1 + sizeof...(TViews)>::type());
	}

protected:


	CUGIP_HD_WARNING_DISABLE
	template<size_t ...N>
	CUGIP_DECL_HYBRID
	value_type call(coord_t aIndex, Sizes<N...> aIndices) const
	{
		return mOperator(mViews.get<N>()[aIndex]...);
	}

	TOperator mOperator;
	Tuple<TView, TViews...> mViews;
};


CUGIP_DECLARE_VIEW_TRAITS(
	(NAryOperatorDeviceImageView<TOperator, TView, TViews...>),
	dimension<TView>::value,
	(MultiViewTraits<TView, TViews...>::cIsDeviceView),
	(MultiViewTraits<TView, TViews...>::cIsHostView),
	typename TOperator, typename TView, typename... TViews);


template<typename TFunctor, typename TView, typename... TViews>
NAryOperatorDeviceImageView<TFunctor, TView, TViews...>
nAryOperator(TFunctor functor, TView view, TViews... views) {
	return NAryOperatorDeviceImageView<TFunctor, TView, TViews...>(functor, view, views...);
}

template<typename TView, typename... TViews>
NAryOperatorDeviceImageView<ZipValues, TView, TViews...>
zipViews(TView view, TViews... views) {
	return NAryOperatorDeviceImageView<ZipValues, TView, TViews...>(ZipValues(), view, views...);
}


template<typename TView, int tIndex>
class AccessDimensionImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		AccessDimensionImageView<TView, tIndex>>
{
public:
	typedef typename TView::extents_t extents_t;
	typedef typename TView::coord_t coord_t;
	typedef typename TView::diff_t diff_t;
	typedef AccessDimensionImageView<TView, tIndex> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(get<tIndex>(std::declval<TView>()[coord_t()])) result_type;
	typedef result_type value_type;
	typedef const result_type const_value_type;
	typedef result_type accessed_type;

	AccessDimensionImageView(TView view) :
		predecessor_type(view.dimensions()),
		mView(view)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	value_type operator[](coord_t index) const {
		return get<tIndex>(mView[index]);
	}

protected:
	TView mView;
};

CUGIP_DECLARE_VIEW_TRAITS(
	(AccessDimensionImageView<TView, tIndex>),
	dimension<TView>::value,
	(is_device_view<TView>::value),
	(is_host_view<TView>::value),
	typename TView, int tIndex);



/// Creates view which returns squared values from the original view
template<typename TView, typename TDimension>
AccessDimensionImageView<TView, TDimension::value>
getDimension(TView view, TDimension /*aIndex*/) {
	return AccessDimensionImageView<TView, TDimension::value>(view);
}

} // namespace cugip
