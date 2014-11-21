#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/kernel_execution_utils.hpp>
#include <cugip/access_utils.hpp>

#include <cugip/neighborhood.hpp>

#include <cugip/static_memory_block.hpp>


#define BLOCK_NLOPT	12
#define BLOCK_WEIGHTS	8

namespace cugip {

namespace detail {

template<typename TMemoryBlock, typename TCoordinates, typename TLocator, typename TParameters>
CUGIP_DECL_DEVICE void
loadData(TMemoryBlock &aData, TLocator aLocator, const TCoordinates &aPaddedCoords, const TCoordinates &aLocalCoords, TParameters aParameters)
{
	typename TMemoryBlock::extents_t extents = TMemoryBlock::dimensions();
	TCoordinates blockExtents(blockDim.x, blockDim.y, blockDim.z);

	TCoordinates counts;
	for (int i = 0; i < 3; ++i) {
		counts[i] = (extents[i] + blockExtents[i] - 1) / blockExtents[i];
	}

	for (int k = 0; k < counts[2]; ++k) {
		TCoordinates offset(0, 0, k * blockExtents[2]);
		for (int j = 0; j < counts[1]; ++j) {
			offset[1] = j * blockExtents[1];
			for (int i = 0; i < counts[0]; ++i) {
				offset[0] = i * blockExtents[0];
				TCoordinates coords = aLocalCoords + offset;
				if (coords < extents) {
					aData[coords] = aLocator[coords];
				}
			}
		}
	}
	/*const size_t cBorder = TParameters::patch_radius + TParameters::search_radius;
	aData[aPaddedCoords] = aLocator[aPaddedCoords];
	typename TMemoryBlock::extents_t extents = TMemoryBlock::dimensions();
	TCoordinates blockExtents(blockDim.x, blockDim.y, blockDim.z);

	// 6 box sides
	for (int i = 0; i < 3; ++i) {
		int idx = aLocalCoords[i];
		while (idx < cBorder) {
			TCoordinates coordTop = aPaddedCoords;
			coordTop[i] = idx;
			aData[coordTop] = aLocator[coordTop];
			coordTop[i] = extents[i] - idx - 1;
			aData[coordTop] = aLocator[coordTop];
			idx += blockExtents[i];
		}
	}

	for (int i = 0; i < 3; ++i) {
		TCoordinates coordTop = aPaddedCoords;
		for (int j = 0; j < 3; ++j) {
			if (i == j) {
				continue;
			}
			coordTop[j] = aLocalCoords[j];
		}
	}*/
	/*for (int i = 0; i < 2; ++i) {
		int idx = aLocalCoords[i];
		while (idx < cBorder) {
			TCoordinates coordTop = aPaddedCoords;
			coordTop[i] = idx;
			aData[coordTop] = aLocator[coordTop];
			coordTop[i] = extents[i] - idx - 1;
			aData[coordTop] = aLocator[coordTop];
			idx += blockExtents[i];
		}
	}*/
}

template<int tPatchRadius>
struct compute_weight
{
	template <typename TMemoryBlock, typename TCoordinates>
	CUGIP_DECL_DEVICE static float
	run(const TMemoryBlock &aData, const TCoordinates &aCoord1, const TCoordinates &aCoord2)
	{
		typedef simple_vector<int, 3> coord_t;
		float weight = 0;
		for(int k = -tPatchRadius; k <= tPatchRadius; ++k) {
			for(int j = -tPatchRadius; j <= tPatchRadius; ++j) {
				for(int i = -tPatchRadius; i <= tPatchRadius; ++i) {
					coord_t offset(i, j, k);
					float diff = aData[aCoord1 + offset] - aData[aCoord2 + offset];
					weight += diff*diff;
				}
			}
		}
		return exp(-weight / 0.5);
	}
};



//tPatchRadius <= 2 int tSearchRadius <= 4
template <typename TInImageView, typename TOutImageView, typename TParameters>
CUGIP_GLOBAL void
kernel_nonlocal_means(TInImageView aIn, TOutImageView aOut, TParameters aParameters)
{
	const int cBorder = TParameters::patch_radius + TParameters::search_radius;
	const int cPatchRadius = TParameters::patch_radius;
	const int cSearchRadius = TParameters::search_radius;
	typedef typename TInImageView::value_type value_type;
	typedef size_traits_3d<8 + 2*(cBorder), 8 + 2*(cBorder), 4 + 2*(cBorder)> shared_data_size;
	typedef static_memory_block<value_type, shared_data_size> MemoryBlock;

	typedef simple_vector<int, 3> coord_t;

	typename TInImageView::extents_t extents = aIn.dimensions();
	coord_t coords = coords_from_block_dim<coord_t>();
	coord_t blockCornerCoords = corner_coords_from_block_dim<coord_t>();
	coord_t localCoords(threadIdx.x, threadIdx.y, threadIdx.z);
	coord_t paddingOffset = coord_t::fill(cBorder);
	coord_t paddedCoords = localCoords + paddingOffset;
	coord_t searchedBlockCornerCoords = blockCornerCoords - paddingOffset;

	CUGIP_SHARED MemoryBlock data;
	loadData(data, aIn.template locator<border_handling_mirror_t>(searchedBlockCornerCoords), paddedCoords, localCoords, aParameters);
	__syncthreads();

	bool use = !(threadIdx.x || threadIdx.y || threadIdx.z);
	if (coords < extents) {
		float acc = 0;
		float value = 0;
		for(int k = -cSearchRadius; k <= cSearchRadius; ++k) {
			/*if (use) {
				printf("AACCC %d\n", k);
			}*/

			for(int j = -cSearchRadius; j <= cSearchRadius; ++j) {
				for(int i = -cSearchRadius; i <= cSearchRadius; ++i) {

					coord_t offset(i, j, k);
					coord_t sampleCoords = paddedCoords + offset;
					float weight = compute_weight<TParameters::patch_radius>::run(data, paddedCoords, sampleCoords);
					value += weight * data[sampleCoords];
					acc += weight;
				}
			}
		}
		/*if (use) {
			printf("AA %d %f %f\n", cSearchRadius, acc, value);
		}*/
		aOut[coords] = value / acc;
		/*if (!(threadIdx.x || threadIdx.y || threadIdx.z)) {
			printf("AAA %d %d %d %f %f\n", coords[0], coords[1], coords[2], aIn[coords], aOut[coords]);
		}*/
	}
}

} // namespace detail

template <int tPatchRadius, int tSearchRadius>
struct nl_means_parameters
{
	static const int patch_radius = tPatchRadius;
	static const int search_radius = tSearchRadius;
};


template <typename TInImageView, typename TOutImageView, typename TParameters>
void
nonlocal_means(TInImageView aIn, TOutImageView aOut, TParameters aParameters)
{
	D_PRINT("nonlocal_means ...");

	dim3 blockSize(8, 8, 4);
	dim3 gridSize = compute_grid_size(blockSize, aIn.dimensions());;

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	cugip::detail::kernel_nonlocal_means<TInImageView, TOutImageView, TParameters>
		<<<gridSize, blockSize>>>(aIn, aOut, aParameters);

	D_PRINT("nonlocal_means done!");
}

}//namespace cugip
