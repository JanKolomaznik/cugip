#pragma once

#include <cugip/math.hpp>
#include <cugip/functors.hpp>
#include <cugip/tuple.hpp>
#include <cugip/detail/view_declaration_utils.hpp>

namespace cugip {

template<int tDimension>
class device_image_view_base
{
public:
	typedef typename dim_traits<tDimension>::extents_t extents_t;

	device_image_view_base(extents_t dimensions)
		: mDimensions(dimensions)
	{}

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mDimensions; }

	extents_t mDimensions;
};

template<int tDimension, typename TDerived>
class device_image_view_crtp
{
public:
	typedef typename dim_traits<tDimension>::extents_t extents_t;
	typedef typename dim_traits<tDimension>::coord_t coord_t;
	typedef typename dim_traits<tDimension>::diff_t diff_t;

	device_image_view_crtp(extents_t dimensions)
		: mDimensions(dimensions)
	{}

	template<typename TBorderHandling>
	CUGIP_DECL_HYBRID image_locator<TDerived, TBorderHandling>
	locator(coord_t aCoordinates) const
	{
		return image_locator<TDerived, TBorderHandling>(*const_cast<TDerived *>(static_cast<const TDerived *>(this)), aCoordinates);
	}

	CUGIP_DECL_HYBRID extents_t
	dimensions() const
	{ return mDimensions; }

	extents_t mDimensions;
};

template<int tDim>
struct dimension<device_image_view_base<tDim>>: dimension_helper<tDim> {};

#define CUGIP_VIEW_TYPEDEFS_VALUE(ElementType, aDimension)\
	typedef typename dim_traits<aDimension>::extents_t extents_t;\
	typedef typename dim_traits<aDimension>::coord_t coord_t;\
	typedef typename dim_traits<aDimension>::diff_t diff_t;\
	typedef ElementType value_type;\
	typedef const ElementType const_value_type;\
	typedef value_type accessed_type;

#define CUGIP_VIEW_TYPEDEFS_REFERENCE(ElementType, aDimension)\
	typedef typename dim_traits<aDimension>::extents_t extents_t;\
	typedef typename dim_traits<aDimension>::coord_t coord_t;\
	typedef typename dim_traits<aDimension>::diff_t diff_t;\
	typedef ElementType value_type;\
	typedef const ElementType const_value_type;\
	typedef value_type &accessed_type;

/// Procedural image view, which returns same value for all indices.
template<typename TElement, int tDimension>
class ConstantDeviceImageView : public device_image_view_base<tDimension> {
public:
	CUGIP_VIEW_TYPEDEFS_VALUE(TElement, tDimension)
	typedef ConstantDeviceImageView<TElement, tDimension> this_t;
	typedef device_image_view_base<tDimension> predecessor_type;

	ConstantDeviceImageView(TElement element, extents_t size) :
		predecessor_type(size),
		mElement(element)
	{}

	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t /*index*/) const {
		return mElement;
	}

protected:
	value_type mElement;
};

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((ConstantDeviceImageView<TElement, tDim>), tDim, typename TElement, int tDim);

/// Utility function to create ConstantDeviceImageView without the need to specify template parameters.
template<typename TElement, int tDimension>
ConstantDeviceImageView<TElement, tDimension>
constantImage(TElement value, simple_vector<int, tDimension> size)
{
	return ConstantDeviceImageView<TElement, tDimension>(value, size);
}

/// Procedural image view, which generates checker board like image.
template<typename TElement, int tDimension>
class CheckerBoardDeviceImageView : public device_image_view_base<tDimension> {
public:
	CUGIP_VIEW_TYPEDEFS_VALUE(TElement, tDimension)
	typedef CheckerBoardDeviceImageView<TElement, tDimension> this_t;
	typedef device_image_view_base<tDimension> predecessor_type;

	CheckerBoardDeviceImageView(TElement white, TElement black, extents_t tile_size, extents_t size)
		: predecessor_type(size)
		, mTileSize(tile_size)
		, mWhite(white)
		, mBlack(black)
	{}

	CUGIP_DECL_HYBRID
	TElement operator[](coord_t index) const
	{
		return sum(div(index, mTileSize)) % 2 ? mWhite : mBlack;
	}

protected:
	extents_t mTileSize;
	value_type mWhite;
	value_type mBlack;
};

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((CheckerBoardDeviceImageView<TElement, tDim>), tDim, typename TElement, int tDim);

template<int tDimension>
class UniqueIdDeviceImageView
	: public device_image_view_crtp<
		tDimension,
		UniqueIdDeviceImageView<tDimension>>
{
public:
	typedef UniqueIdDeviceImageView<tDimension> this_t;
	typedef device_image_view_crtp<tDimension, this_t> predecessor_type;
	typedef typename predecessor_type::coord_t coord_t;
	typedef typename predecessor_type::extents_t extents_t;
	typedef int value_type;
	typedef const int const_value_type;
	typedef int accessed_type;

	UniqueIdDeviceImageView(extents_t aSize)
		: predecessor_type(aSize)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	value_type operator[](coord_t index) const {
		//return get_zorder_access_index(this->dimensions(), index) + 1;
		//return get_blocked_order_access_index(this->dimensions(), index) + 1;
		return get_linear_access_index(this->dimensions(), index) + 1;
	}

protected:
};

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((UniqueIdDeviceImageView<tDimension>), tDimension, int tDimension);


/// Utility function to create CheckerBoardDeviceImageView without the need to specify template parameters.
template<typename TElement, int tDimension>
CheckerBoardDeviceImageView<TElement, tDimension>
checkerBoard(
			TElement white,
			TElement black,
			const simple_vector<int, tDimension> &tile_size,
			const simple_vector<int, tDimension> &size)
{
	return CheckerBoardDeviceImageView<TElement, tDimension>(white, black, tile_size, size);
}


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
class LinearCombinationDeviceImageView : public BinaryOperatorDeviceImageView<TView1, TView2> {
public:
	typedef typename TView1::extents_t extents_t;
	typedef typename TView1::coord_t coord_t;
	typedef typename TView1::diff_t diff_t;
	typedef BinaryOperatorDeviceImageView<TView1, TView2> predecessor_type;

	typedef typename TView1::value_type value1_type;
	typedef typename TView2::value_type value2_type;


	typedef typename std::common_type<TFactor1, value1_type>::type result1_type;
	typedef typename std::common_type<TFactor2, value2_type>::type result2_type;
	typedef typename std::common_type<result1_type, result2_type>::type value_type;
	typedef const value_type const_value_type;
	typedef value_type accessed_type;

	LinearCombinationDeviceImageView(TFactor1 factor1, TView1 view1, TFactor2 factor2, TView2 view2) :
		predecessor_type(view1, view2),
		mFactor1(factor1),
		mFactor2(factor2)
	{}

	CUGIP_DECL_DEVICE
	value_type operator[](coord_t index) const {
		return mFactor1 * this->mView1[index] + mFactor2 * this->mView2[index];
	}

protected:
	TFactor1 mFactor1;
	TFactor2 mFactor2;
};

CUGIP_DECLARE_DEVICE_VIEW_TRAITS((LinearCombinationDeviceImageView<TFactor1, TView1, TFactor2, TView2>), dimension<TView1>::value, typename TFactor1, typename TView1, typename TFactor2, typename TView2);

/// Utility function to create LinearCombinationDeviceImageView without the need to specify template parameters.
template<typename TFactor1, typename TView1, typename TFactor2, typename TView2>
LinearCombinationDeviceImageView<TFactor1, TView1, TFactor2, TView2>
linearCombination(TFactor1 factor1, TView1 view1, TFactor2 factor2, TView2 view2) {
	return LinearCombinationDeviceImageView<TFactor1, TView1, TFactor2, TView2>(factor1, view1, factor2, view2);
}


/// Utility function wrapping two image addition.
template<typename TView1, typename TView2>
LinearCombinationDeviceImageView<int, TView1, int, TView2>
add(TView1 view1, TView2 view2) {
	// TODO(johny) - possible more efficient implementation
	return LinearCombinationDeviceImageView<int, TView1, int, TView2>(1, view1, 1, view2);
}


/// Utility function wrapping two image subtraction.
template<typename TView1, typename TView2>
LinearCombinationDeviceImageView<int, TView1, int, TView2>
subtract(TView1 view1, TView2 view2) {
	// TODO(johny) - possible more efficient implementation
	return LinearCombinationDeviceImageView<int, TView1, int, TView2>(1, view1, -1, view2);
}


/// Image view, which returns per element multiplication.
/// R = I1 .* I2
template<typename TView1, typename TView2>
class MultiplicationDeviceImageView : public BinaryOperatorDeviceImageView<TView1, TView2> {
public:
	typedef simple_vector<int, dimension<TView1>::value> extents_t;
	typedef simple_vector<int, dimension<TView1>::value> coord_t;
	typedef BinaryOperatorDeviceImageView<TView1, TView2> predecessor_type;

	typedef typename TView1::Element Element1;
	typedef typename TView2::Element Element2;


	typedef decltype(std::declval<Element1>() * std::declval<Element2>()) Element;
	typedef Element AccessType;

	MultiplicationDeviceImageView(TView1 view1, TView2 view2) :
		predecessor_type(view1, view2)
	{}

	CUGIP_DECL_DEVICE
	Element operator[](coord_t index) const {
		return this->view1_[index] * this->view2_[index];
	}
};


/// Utility function to create MultiplicationDeviceImageView without the need to specify template parameters.
template<typename TView1, typename TView2>
MultiplicationDeviceImageView<TView1, TView2>
multiply(TView1 view1, TView2 view2) {
	return MultiplicationDeviceImageView<TView1, TView2>(view1, view2);
}

/// Image view, which returns per element division.
/// R = I1 ./ I2
template<typename TView1, typename TView2>
class DivisionDeviceImageView : public BinaryOperatorDeviceImageView<TView1, TView2> {
public:
	static const bool kIsDeviceView = true;
	static const int kDimension = TView1::kDimension;
	typedef simple_vector<int, kDimension> extents_t;
	typedef simple_vector<int, kDimension> coord_t;
	typedef BinaryOperatorDeviceImageView<TView1, TView2> predecessor_type;

	typedef typename TView1::Element Element1;
	typedef typename TView2::Element Element2;


	typedef typename std::common_type<Element1, Element2>::type Element;
	typedef Element AccessType;

	DivisionDeviceImageView(TView1 view1, TView2 view2) :
		predecessor_type(view1, view2)
	{}

	CUGIP_DECL_DEVICE
	Element operator[](coord_t index) const {
		return this->view1_[index] / this->view2_[index];
	}
};


/// Utility function to create DivisionDeviceImageView without the need to specify template parameters.
template<typename TView1, typename TView2>
DivisionDeviceImageView<TView1, TView2>
divide(TView1 view1, TView2 view2) {
	return DivisionDeviceImageView<TView1, TView2>(view1, view2);
}


/// View which allows mirror access to another view
/// TODO(johny) - specialization for memory based views - only stride and pointer reordering
template<typename TView>
class MirrorDeviceImageView : public device_image_view_base<TView::kDimension> {
public:
	static const bool kIsDeviceView = true;
	static const int kDimension = TView::kDimension;
	typedef typename TView::extents_t extents_t;
	typedef typename TView::coord_t coord_t;
	typedef device_image_view_base<TView::kDimension> predecessor_type;
	typedef typename TView::Element Element;
	typedef typename TView::AccessType AccessType;

	MirrorDeviceImageView(TView view, simple_vector<bool, kDimension> flips) :
		predecessor_type(view.dimensions()),
		view_(view),
		flips_(flips)
	{}

	CUGIP_DECL_DEVICE
	AccessType operator[](coord_t index) const {
		return view_[FlipCoordinates(index, this->dimensions(), flips_)];
	}

protected:
	CUGIP_DECL_HYBRID
	static coord_t FlipCoordinates(coord_t index, extents_t size, simple_vector<bool, kDimension> flips) {
		for (int i = 0; i < kDimension; ++i) {
			if (flips[i]) {
				index[i] = size[i] - index[i] - 1;
			}
		}
		return index;
	}

	TView view_;
	simple_vector<bool, kDimension> flips_;
};


/// Create mirror views with fliped axes specification
template<typename TView>
MirrorDeviceImageView<TView>
mirror(TView view, simple_vector<bool, TView::kDimension> flips) {
	return MirrorDeviceImageView<TView>(view, flips);
}


/// View which pads another image view
/// TODO(johny) - do also tapper padding
template<typename TView/*, bool tIsPeriodic*/>
class PaddedDeviceImageView : public device_image_view_base<TView::kDimension> {
public:
	static const bool kIsDeviceView = true;
	static const int kDimension = TView::kDimension;
	typedef typename TView::extents_t extents_t;
	typedef typename TView::coord_t coord_t;
	typedef device_image_view_base<TView::kDimension> predecessor_type;
	typedef typename TView::Element Element;
	typedef typename TView::Element AccessType;

	PaddedDeviceImageView(TView view, const extents_t &size, const extents_t &offset, Element fill_value) :
		predecessor_type(size),
		view_(view),
		offset_(offset),
		fill_value_(fill_value)
	{}

	CUGIP_DECL_DEVICE
	AccessType operator[](coord_t index) const {
		index = ModPeriodic(index - offset_, this->dimensions());
		if (view_.IsIndexInside(index)) {
			return view_[index];
		}
		return fill_value_;
	}

protected:
	TView view_;
	simple_vector<int, kDimension> offset_;
	Element fill_value_;
};


/// Create padded view
template<typename TView>
PaddedDeviceImageView<TView>
padView(
	TView view,
	simple_vector<int, TView::kDimension> size,
	simple_vector<int, TView::kDimension> offset,
	typename TView::Element fill_value)
{
	return PaddedDeviceImageView<TView>(view, size, offset, fill_value);
}


/// View which allows mirror access to another view
/// TODO(johny) - specialization for memory based views - only stride and pointer reordering
template<typename TView, typename TOperator>
class UnaryOperatorDeviceImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		UnaryOperatorDeviceImageView<TView, TOperator>>
{
public:
	typedef typename TView::extents_t extents_t;
	typedef typename TView::coord_t coord_t;
	typedef typename TView::diff_t diff_t;
	typedef UnaryOperatorDeviceImageView<TView, TOperator> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(std::declval<TOperator>()(std::declval<Input>())) result_type;
	typedef result_type value_type;
	typedef const result_type const_value_type;
	typedef result_type accessed_type;

	UnaryOperatorDeviceImageView(TView view, TOperator unary_operator) :
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

//CUGIP_DECLARE_DEVICE_VIEW_TRAITS((UnaryOperatorDeviceImageView<TView, TOperator>), dimension<TView>::value, typename TView, typename TOperator);
CUGIP_DECLARE_HYBRID_VIEW_TRAITS((UnaryOperatorDeviceImageView<TView, TOperator>), dimension<TView>::value, typename TView, typename TOperator);

/// Creates view which returns squared values from the original view
template<typename TView>
UnaryOperatorDeviceImageView<TView, SquareFunctor>
square(TView view) {
	return UnaryOperatorDeviceImageView<TView, SquareFunctor>(view, SquareFunctor());
}


/// Creates view which returns square root of the values from the original view
template<typename TView>
UnaryOperatorDeviceImageView<TView, SquareRootFunctor>
squareRoot(TView view) {
	return UnaryOperatorDeviceImageView<TView, SquareRootFunctor>(view, SquareRootFunctor());
}


/// Utility function to create multiplied view without the need to specify template parameters.
template<typename TFactor, typename TView>
UnaryOperatorDeviceImageView<TView, MultiplyByFactorFunctor<TFactor>>
multiplyByFactor(TFactor factor, TView view) {
	return UnaryOperatorDeviceImageView<TView, MultiplyByFactorFunctor<TFactor>>(view, MultiplyByFactorFunctor<TFactor>(factor));
}

/// Creates view returning values from the original view with value added.
template<typename TType, typename TView>
UnaryOperatorDeviceImageView<TView, AddValueFunctor<TType>>
addValue(TType value, TView view) {
	return UnaryOperatorDeviceImageView<TView, AddValueFunctor<TType>>(view, AddValueFunctor<TType>(value));
}


/// Returns view returning values from the original view with values lower then limit replaced by the limit
template<typename TType, typename TView>
UnaryOperatorDeviceImageView<TView, MaxFunctor<TType>>
lowerLimit(TType limit, TView view) {
	return UnaryOperatorDeviceImageView<TView, MaxFunctor<TType>>(view, MaxFunctor<TType>(limit));
}


/// Returns view returning values from the original view with values lower then limit replaced by the specified replacement
template<typename TType, typename TView>
UnaryOperatorDeviceImageView<TView, LowerLimitFunctor<TType>>
lowerLimit(TType limit, TType replacement, TView view) {
	return UnaryOperatorDeviceImageView<TView, LowerLimitFunctor<TType>>(view, LowerLimitFunctor<TType>(limit, replacement));
}


/// Returns view returning values from the original view with values bigger then limit replaced by the limit
template<typename TType, typename TView>
UnaryOperatorDeviceImageView<TView, MinFunctor<TType>>
upperLimit(TType limit, TView view) {
	return UnaryOperatorDeviceImageView<TView, MinFunctor<TType>>(view, MinFunctor<TType>(limit));
}


/// Returns view returning values from the original view with values bigger then limit replaced by the specified replacement
template<typename TType, typename TView>
UnaryOperatorDeviceImageView<TView, UpperLimitFunctor<TType>>
upperLimit(TType limit, TType replacement, TView view) {
	return UnaryOperatorDeviceImageView<TView, UpperLimitFunctor<TType>>(view, UpperLimitFunctor<TType>(limit, replacement));
}

template<typename TFunctor, typename TView>
UnaryOperatorDeviceImageView<TView, TFunctor>
unaryOperator(TView view, TFunctor functor) {
	return UnaryOperatorDeviceImageView<TView, TFunctor>(view, functor);
}

template<typename TView, typename TOperator>
class UnaryOperatorOnPositionDeviceImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		UnaryOperatorOnPositionDeviceImageView<TView, TOperator>>
{
public:
	typedef typename TView::extents_t extents_t;
	typedef typename TView::coord_t coord_t;
	typedef typename TView::diff_t diff_t;
	typedef UnaryOperatorOnPositionDeviceImageView<TView, TOperator> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(std::declval<TOperator>()(std::declval<Input>(), coord_t())) result_type;
	typedef result_type value_type;
	typedef const result_type const_value_type;
	typedef result_type accessed_type;

	UnaryOperatorOnPositionDeviceImageView(TView view, TOperator unary_operator) :
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

//CUGIP_DECLARE_DEVICE_VIEW_TRAITS((UnaryOperatorDeviceImageView<TView, TOperator>), dimension<TView>::value, typename TView, typename TOperator);
CUGIP_DECLARE_HYBRID_VIEW_TRAITS((UnaryOperatorOnPositionDeviceImageView<TView, TOperator>), dimension<TView>::value, typename TView, typename TOperator);


template<typename TFunctor, typename TView>
UnaryOperatorOnPositionDeviceImageView<TView, TFunctor>
unaryOperatorOnPosition(TView view, TFunctor functor) {
	return UnaryOperatorOnPositionDeviceImageView<TView, TFunctor>(view, functor);
}

template<typename TView, typename TOperator, typename TBorderHandling = cugip::border_handling_repeat_t>
class UnaryOperatorOnLocatorDeviceImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		UnaryOperatorOnLocatorDeviceImageView<TView, TOperator, TBorderHandling>>
{
public:
	typedef typename TView::extents_t extents_t;
	typedef typename TView::coord_t coord_t;
	typedef typename TView::diff_t diff_t;
	typedef UnaryOperatorOnLocatorDeviceImageView<TView, TOperator, TBorderHandling> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(std::declval<TOperator>()(std::declval<image_locator<TView, TBorderHandling>>())) result_type;
	typedef result_type value_type;
	typedef const result_type const_value_type;
	typedef result_type accessed_type;
	//TODO - support border handling
	UnaryOperatorOnLocatorDeviceImageView(TView view, TOperator unary_operator) :
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

//CUGIP_DECLARE_DEVICE_VIEW_TRAITS((UnaryOperatorDeviceImageView<TView, TOperator>), dimension<TView>::value, typename TView, typename TOperator);
CUGIP_DECLARE_HYBRID_VIEW_TRAITS((UnaryOperatorOnLocatorDeviceImageView<TView, TOperator>), dimension<TView>::value, typename TView, typename TOperator);


template<typename TFunctor, typename TView>
UnaryOperatorOnLocatorDeviceImageView<TView, TFunctor>
unaryOperatorOnLocator(TView view, TFunctor functor) {
	return UnaryOperatorOnLocatorDeviceImageView<TView, TFunctor>(view, functor);
}
/// View returning single coordinate mapping from grid
/// Inspired by Matlab function 'meshgrid'
template<int tDimension>
class MeshGridView: public device_image_view_base<tDimension> {
public:
	typedef simple_vector<int, tDimension> extents_t;
	typedef simple_vector<int, tDimension> coord_t;
	typedef device_image_view_base<tDimension> predecessor_type;
	typedef int value_type;
	typedef value_type access_type;

	MeshGridView() :
		predecessor_type(extents_t()),
		mDimension(0),
		mStart(0),
		mIncrement(0)
	{}

	MeshGridView(coord_t aFrom, coord_t aTo, int dimension) :
		predecessor_type(abs(aTo - aFrom)),
		mDimension(dimension),
		mStart(aFrom[dimension]),
		mIncrement(signum(aTo[dimension] - aFrom[dimension]))
	{}

	CUGIP_DECL_DEVICE
	access_type operator[](coord_t index) const {
		return mStart + index[mDimension] * mIncrement;
	}

protected:
	int mDimension;
	int mStart;
	int mIncrement;
};

/// Rectangular grid in N-D space.
/// Inspired by Matlab function of the same name:
/// [X,Y] = MeshGrid(Int2(1, 10), Int2(3, 14))
/// X =
///      1     2     3
///      1     2     3
///      1     2     3
///      1     2     3
///      1     2     3
/// Y =
///     10    10    10
///     11    11    11
///     12    12    12
///     13    13    13
///     14    14    14
template<int tDimension>
std::array<MeshGridView<tDimension>, tDimension>
meshGrid(simple_vector<int, tDimension> from, simple_vector<int, tDimension> to) {
	std::array<MeshGridView<tDimension>, tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = MeshGridView<tDimension>(from, to, i);
	}
	return result;
}

template<typename TFirstView, typename... TViews>
struct MultiViewTraits
{
	static const int cDimension = dimension<TFirstView>::value;

	typedef typename TFirstView::extents_t extents_t;
	typedef typename TFirstView::coord_t coord_t;
	typedef typename TFirstView::diff_t diff_t;
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
	typedef MultiViewTraits<TViews...> MultiTraits;

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


CUGIP_DECLARE_HYBRID_VIEW_TRAITS((NAryOperatorDeviceImageView<TOperator, TView, TViews...>), dimension<TView>::value, typename TOperator, typename TView, typename... TViews);


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
class AccessDimensionDeviceImageView
	: public device_image_view_crtp<
		dimension<TView>::value,
		AccessDimensionDeviceImageView<TView, tIndex>>
{
public:
	typedef typename TView::extents_t extents_t;
	typedef typename TView::coord_t coord_t;
	typedef typename TView::diff_t diff_t;
	typedef AccessDimensionDeviceImageView<TView, tIndex> this_t;
	typedef device_image_view_crtp<dimension<TView>::value, this_t> predecessor_type;
	typedef typename TView::value_type Input;
	typedef decltype(get<tIndex>(std::declval<TView>()[coord_t()])) result_type;
	typedef result_type value_type;
	typedef const result_type const_value_type;
	typedef result_type accessed_type;

	AccessDimensionDeviceImageView(TView view) :
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

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((AccessDimensionDeviceImageView<TView, tIndex>), dimension<TView>::value, typename TView, int tIndex);

/// Creates view which returns squared values from the original view
template<typename TView, typename TDimension>
AccessDimensionDeviceImageView<TView, TDimension::value>
getDimension(TView view, TDimension /*aIndex*/) {
	return AccessDimensionDeviceImageView<TView, TDimension::value>(view);
}


} // namespace cugip
