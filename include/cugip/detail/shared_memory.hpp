#pragma once

#include <cugip/static_int_sequence.hpp>
#include <cugip/image_view.hpp>
#include <cugip/cuda_utils.hpp>

namespace cugip {


namespace detail {

struct RunTimeThreadBlockSize
{
	CUGIP_DECL_HYBRID
	static int count() {
		return blockDim.x * blockDim.y * blockDim.z;
	}
};

template<typename TElement, typename TStaticSize, typename TThreadBlockSize = RunTimeThreadBlockSize>
class SharedMemory
{
public:
	static constexpr int cDimension = TStaticSize::cDimension;
	static constexpr int cElementCount = TStaticSize::count();
	static constexpr int cBufferSize = cElementCount * sizeof(TElement);
	static constexpr int cLayerSize = TStaticSize::count() / TStaticSize::last();

	typedef simple_vector<int, cDimension> coord_t;

	template<typename TView>
	CUGIP_DECL_DEVICE
	void load(TView aView, coord_t aCorner/*region<cDimension> aRegion*/)
	{
		int index = threadOrderFromIndex();
		TElement *buffer = reinterpret_cast<TElement *>(data);
		while (index < cElementCount) {
			auto coords = min_coords(
				aView.dimensions()-coord_t(1, FillFlag()),
				max_coords(coord_t(),
					aCorner + index_from_linear_access_index(to_vector(TStaticSize()), index)));
			buffer[index] = aView[coords];

			index += TThreadBlockSize::count();
		}
	}

	template<typename TView>
	CUGIP_DECL_DEVICE
	void loadZeroOut(TView aView, coord_t aCorner/*region<cDimension> aRegion*/)
	{
		int index = threadOrderFromIndex();
		TElement *buffer = reinterpret_cast<TElement *>(data);
		while (index < cElementCount) {
			auto uncheckedCoordinates = aCorner + index_from_linear_access_index(to_vector(TStaticSize()), index);
			auto coords = min_coords(
				aView.dimensions()-coord_t(1, FillFlag()),
				max_coords(coord_t(), uncheckedCoordinates));

			if (coords != uncheckedCoordinates) {
				buffer[index] = 0;
			} else {
				buffer[index] = aView[coords];
			}

			index += TThreadBlockSize::count();
		}
	}

	CUGIP_DECL_DEVICE
	TElement &
	get(coord_t aCoords)
	{
		TElement *buffer = reinterpret_cast<TElement *>(data);
		return buffer[get_linear_access_index(to_vector(TStaticSize()), aCoords)];
	}

	CUGIP_DECL_DEVICE
	device_image_view<TElement, cDimension>
	view()
	{
		return makeDeviceImageView(reinterpret_cast<TElement *>(data), to_vector(TStaticSize()));
	}


	CUGIP_DECL_DEVICE
	void shift_up(int aShift)
	{
		int layerCount = TStaticSize::last() - aShift;
		int shiftStride = aShift * cLayerSize;

		TElement *buffer = reinterpret_cast<TElement *>(data);
		for(int i = 0; i < layerCount; ++i) {
			int layerStartIndex = i * cLayerSize;
			int index = threadOrderFromIndex();
			while(index < cLayerSize) {
				buffer[layerStartIndex + index] = buffer[layerStartIndex + shiftStride + index];
				index += TThreadBlockSize::count();
			}
			__syncthreads();
		}
	}

	CUGIP_DECL_DEVICE
	void shift_up2(int aShift)
	{
		int layerCount = TStaticSize::last() - aShift;
		int shiftStride = aShift * cLayerSize;

		int blockSteps = (cLayerSize * layerCount + layerCount - 1) / TThreadBlockSize::count();

		TElement *buffer = reinterpret_cast<TElement *>(data);
		int index = threadOrderFromIndex();
		for (int i = 0; i < blockSteps; ++i) {
			if (index < layerCount * cLayerSize) {
				buffer[index] = buffer[index + shiftStride];
			}
			__syncthreads();
		}
	}

	template<typename TView>
	CUGIP_DECL_DEVICE
	void shift_and_load(TView aView, coord_t aCorner, int aShift) {
		shift_up2(aShift);

		int layerCount = TStaticSize::last() - aShift;
		int index = layerCount * cLayerSize + threadOrderFromIndex();
		TElement *buffer = reinterpret_cast<TElement *>(data);
		while (index < cElementCount) {
			auto coords = min_coords(
				aView.dimensions()-coord_t(1, FillFlag()),
				max_coords(coord_t(),
					aCorner + index_from_linear_access_index(TStaticSize::vector(), index)));
			buffer[index] = aView[coords];

			index += TThreadBlockSize::count();
		}
	}


protected:

	int8_t data[cBufferSize];
};

} // namespace detail
} // namespace cugip
