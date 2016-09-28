#pragma once


#include <cugip/static_int_sequence.hpp>

namespace cugip {

/** \defgroup meta_algorithm
 *
 **/
namespace detail {

#if defined(__CUDACC__)

template<int tDimension>
CUGIP_DECL_HYBRID dim3 defaultBlockDimForDimension();

template<>
inline CUGIP_DECL_HYBRID dim3 defaultBlockDimForDimension<2>() { return dim3(32, 16, 1); }

template<>
inline CUGIP_DECL_HYBRID dim3 defaultBlockDimForDimension<3>() { return dim3(32, 4, 4); }

template<int tDimension>
CUGIP_DECL_HYBRID dim3 defaultGridSizeForBlockDim(cugip::simple_vector<int, tDimension> aViewDimensions, dim3 aBlockSize);

template<>
inline CUGIP_DECL_HYBRID dim3
defaultGridSizeForBlockDim<2>(cugip::simple_vector<int, 2> aViewDimensions, dim3 aBlockSize)
{
	return dim3(
		(aViewDimensions[0] - 1) / aBlockSize.x + 1,
		(aViewDimensions[1] - 1) / aBlockSize.y + 1,
		1);
}

template<>
inline CUGIP_DECL_HYBRID dim3
defaultGridSizeForBlockDim<3>(cugip::simple_vector<int, 3> aViewDimensions, dim3 aBlockSize)
{
	return dim3(
		(aViewDimensions[0] - 1) / aBlockSize.x + 1,
		(aViewDimensions[1] - 1) / aBlockSize.y + 1,
		(aViewDimensions[2] - 1) / aBlockSize.z + 1);
}

template<int tDimension>
struct defaultBlockSize;

template<>
struct defaultBlockSize<2>
{
	typedef StaticSize<32, 16> type;
};

template<>
struct defaultBlockSize<3>
{
	typedef StaticSize<32, 4, 4> type;
};

#endif //defined(__CUDACC__)
} // namespace detail

}//namespace cugip
