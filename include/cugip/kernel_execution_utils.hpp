#pragma once

 #include <boost/type_traits.hpp>
 #include <cugip/traits.hpp>
 #include <cugip/utils.hpp>

namespace cugip {

template<typename TImageSize>
dim3
compute_grid_size(const dim3 &aBlockSize, const TImageSize &aImageSize)
{
	cugip::simple_vector<int, 3> grid(1, 1, 1);
	cugip::simple_vector<int, 3> block(aBlockSize.x, aBlockSize.y, aBlockSize.z);

	for (int i = 0; i < dimension<TImageSize>::value; ++i) {
		grid[i] = (aImageSize[i] + block[i] - 1) / block[i];
	}
	return dim3(grid[0], grid[1], grid[2]);
}

template<typename TCoords>
CUGIP_DECL_DEVICE TCoords
corner_coords_from_block_dim()
{
	cugip::simple_vector<int, 3> index(blockIdx.x, blockIdx.y, blockIdx.z);
	cugip::simple_vector<int, 3> size(blockDim.x, blockDim.y, blockDim.z);
	TCoords result;
	for (int i = 0; i < dimension<TCoords>::value; ++i) {
		result[i] = index[i] * size[i];
	}
	return result;
}

template<typename TCoords>
CUGIP_DECL_DEVICE TCoords
coords_from_block_dim()
{
	cugip::simple_vector<int, 3> thread_index(threadIdx.x, threadIdx.y, threadIdx.z);
	cugip::simple_vector<int, 3> index(blockIdx.x, blockIdx.y, blockIdx.z);
	cugip::simple_vector<int, 3> size(blockDim.x, blockDim.y, blockDim.z);
	TCoords result;
	for (int i = 0; i < dimension<TCoords>::value; ++i) {
		result[i] = index[i] * size[i] + thread_index[i];
	}
	return result;
}




}//namespace cugip

