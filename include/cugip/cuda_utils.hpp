#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>
#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <iostream>

namespace cugip {

/// \param bytes Size of array element in bytes.
/// \param size Size of 2D array in number of elements.
inline cudaExtent makeCudaExtent(int bytes, Int2 size) {
	return make_cudaExtent(bytes * size[0], size[1], 1);
}

/// \param bytes Size of array element in bytes.
/// \param size Size of 3D array in number of elements.
inline cudaExtent makeCudaExtent(int bytes, Int3 size) {
	return make_cudaExtent(bytes * size[0], size[1], size[2]);
}

/// Convert cuda pitched pointer to element based strides.
/// \param bytes Size of element in bytes.
/// \param pitched_ptr Actual pitched pointer.
template<int tDimension>
simple_vector<int, tDimension> pitchedPtrToStrides(int bytes, cudaPitchedPtr pitched_ptr);

template<>
inline Int2 pitchedPtrToStrides<2>(int bytes, cudaPitchedPtr pitched_ptr) {
	//CUGIP_ASSERT(pitched_ptr.pitch % bytes == 0);
	return Int2(bytes, pitched_ptr.pitch);
}

template<>
inline Int3 pitchedPtrToStrides<3>(int bytes, cudaPitchedPtr pitched_ptr) {
	//CUGIP_ASSERT(pitched_ptr.pitch % bytes == 0);
	return Int3(bytes, pitched_ptr.pitch, pitched_ptr.pitch * pitched_ptr.ysize);
}

/// Convert element based strides to cuda pitched pointer.
/// \param ptr Data pointer
/// \param size Size of image buffer
template<typename TElement>
inline cudaPitchedPtr stridesToPitchedPtr(TElement *ptr, Int2 size, Int2 strides) {
	CUGIP_ASSERT(strides[0] == sizeof(TElement) && "Pitched cuda pointer is usable only for continuous mmemory blocks");
	// pitched pointer wraps only void * -> goodbye const correctness here
	return make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(ptr)), strides[1], size[0], size[1]);
}

template<typename TElement>
inline cudaPitchedPtr stridesToPitchedPtr(TElement *ptr, Int3 size, Int3 strides) {
	CUGIP_ASSERT(strides[0] == sizeof(TElement) && "Pitched cuda pointer is usable only for continuous mmemory blocks");
	// pitched pointer wraps only void * -> goodbye const correctness here
	return make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(ptr)), strides[1], size[0], size[1]);
}

/// Get 1 to 1 mapping between cuda threads in grid and view coordinates
template<int tDimension>
CUGIP_DECL_DEVICE cugip::simple_vector<int, tDimension>
mapBlockIdxAndThreadIdxToViewCoordinates();

template<>
CUGIP_DECL_DEVICE cugip::simple_vector<int, 2>
mapBlockIdxAndThreadIdxToViewCoordinates<2>()
{
	CUGIP_ASSERT(gridDim.z == 1);
	CUGIP_ASSERT(blockDim.z == 1);
	CUGIP_ASSERT(threadIdx.z == 0);
	return cugip::simple_vector<int, 2>(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

template<>
CUGIP_DECL_DEVICE cugip::simple_vector<int, 3>
mapBlockIdxAndThreadIdxToViewCoordinates<3>()
{
	return cugip::simple_vector<int, 3>(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
}

CUGIP_DECL_DEVICE
inline int threadOrderFromIndex()
{
	return threadIdx.x
		+ threadIdx.y * blockDim.x
		+ threadIdx.z * blockDim.x * blockDim.y;
}

}//namespace cugip

inline std::ostream &operator<<(std::ostream &stream, const cudaPitchedPtr &pointer) {
	return stream << boost::format("[pitch: %1%; ptr %2%; xsize: %3%; ysize: %4%]")
		% pointer.pitch
		% pointer.ptr
		% pointer.xsize
		% pointer.ysize;
}

inline std::ostream &operator<<(std::ostream &stream, const cudaExtent &extent) {
	return stream << boost::format("[w: %1%; h %2%; d: %3%]")
		% extent.width
		% extent.height
		% extent.depth;
}
