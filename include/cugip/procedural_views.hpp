#pragma once

#include <cugip/math.hpp>
#include <cugip/functors.hpp>
#include <cugip/tuple.hpp>
#include <cugip/detail/view_declaration_utils.hpp>

#include <cugip/image_locator.hpp>

namespace cugip {


/// Procedural image view, which returns same value for all indices.
template<typename TElement, int tDimension>
class ConstantImageView : public hybrid_image_view_base<tDimension> {
public:
	typedef ConstantImageView<TElement, tDimension> this_t;
	typedef hybrid_image_view_base<tDimension> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(TElement, tDimension)

	ConstantImageView(TElement element, extents_t size) :
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

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((ConstantImageView<TElement, tDim>), tDim, typename TElement, int tDim);

/// Utility function to create ConstantDeviceImageView without the need to specify template parameters.
template<typename TElement, int tDimension>
ConstantImageView<TElement, tDimension>
constantImage(TElement value, simple_vector<int, tDimension> size)
{
	return ConstantImageView<TElement, tDimension>(value, size);
}

/// Procedural image view, which generates checker board like image.
template<typename TElement, int tDimension>
class CheckerBoardImageView : public hybrid_image_view_base<tDimension> {
public:
	typedef CheckerBoardImageView<TElement, tDimension> this_t;
	typedef hybrid_image_view_base<tDimension> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(TElement, tDimension)

	CheckerBoardImageView(TElement white, TElement black, extents_t tile_size, extents_t size)
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

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((CheckerBoardImageView<TElement, tDim>), tDim, typename TElement, int tDim);

/// Utility function to create CheckerBoardImageView without the need to specify template parameters.
template<typename TElement, int tDimension>
CheckerBoardImageView<TElement, tDimension>
checkerBoard(
			TElement white,
			TElement black,
			const simple_vector<int, tDimension> &tile_size,
			const simple_vector<int, tDimension> &size)
{
	return CheckerBoardImageView<TElement, tDimension>(white, black, tile_size, size);
}

template<typename TId = int>
struct LinearAccessIndex
{
	typedef TId id_type;

	template<typename TExtents, typename TCoordinates>
	CUGIP_DECL_HYBRID
	static id_type compute(TExtents aSize, TCoordinates aIndex)
	{
		return get_linear_access_index(aSize, aIndex);
	}
};

template<int tBlockSize = 2, typename TId = int>
struct BlockedAccessIndex
{
	typedef TId id_type;

	template<typename TExtents, typename TCoordinates>
	CUGIP_DECL_HYBRID
	static id_type compute(TExtents aSize, TCoordinates aIndex)
	{
		return get_blocked_order_access_index<tBlockSize>(aSize, aIndex);
	}
};

template<int tDimension, typename TIdGenerator = LinearAccessIndex<int>>
class UniqueIdDeviceImageView
	: public device_image_view_crtp<
		tDimension,
		UniqueIdDeviceImageView<tDimension, TIdGenerator>>
{
public:
	//TODO - bigger int
	typedef UniqueIdDeviceImageView<tDimension, TIdGenerator> this_t;
	typedef device_image_view_crtp<tDimension, this_t> predecessor_type;
	typedef typename TIdGenerator::id_type id_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(id_type, tDimension)

	UniqueIdDeviceImageView(extents_t aSize)
		: predecessor_type(aSize)
	{}

	CUGIP_HD_WARNING_DISABLE
	CUGIP_DECL_HYBRID
	value_type operator[](coord_t index) const {
		//return get_zorder_access_index(this->dimensions(), index) + 1;
		//return get_blocked_order_access_index(this->dimensions(), index) + 1;
		//return get_linear_access_index(this->dimensions(), index) + 1;
		return TIdGenerator::compute(this->dimensions(), index) + 1;
	}

protected:
};

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((UniqueIdDeviceImageView<tDimension, TId>), tDimension, int tDimension, typename TId);


/// View returning single coordinate mapping from grid
/// Inspired by Matlab function 'meshgrid'
template<int tDimension>
class CoordinatesView: public hybrid_image_view_base<tDimension> {
public:
	typedef hybrid_image_view_base<tDimension> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(coord_t, tDimension)

	CoordinatesView() :
		predecessor_type(extents_t())
	{}

	CoordinatesView(extents_t aSize) :
		predecessor_type(aSize)
	{}

	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return index;
	}

};

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((CoordinatesView<tDimension>), tDimension, int tDimension);



/// View returning single coordinate mapping from grid
/// Inspired by Matlab function 'meshgrid'
template<int tDimension>
class MeshGridView: public hybrid_image_view_base<tDimension> {
public:
	typedef hybrid_image_view_base<tDimension> predecessor_type;
	CUGIP_VIEW_TYPEDEFS_VALUE(int, tDimension)

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

	CUGIP_DECL_HYBRID
	accessed_type operator[](coord_t index) const {
		return mStart + index[mDimension] * mIncrement;
	}

protected:
	int mDimension;
	int mStart;
	int mIncrement;
};

CUGIP_DECLARE_HYBRID_VIEW_TRAITS((MeshGridView<tDimension>), tDimension, int tDimension);

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


} // namespace cugip
