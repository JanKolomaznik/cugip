#pragma once

#include <cugip/static_int_sequence.hpp>

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

	typedef simple_vector<int, cDimension> coord_t;

	template<typename TView>
	CUGIP_DECL_DEVICE
	void load(TView aView, coord_t aCorner/*region<cDimension> aRegion*/)
	{
		int index = threadOrderFromIndex();
		TElement *buffer = reinterpret_cast<TElement *>(data);
		while (index < cElementCount) {
			auto coords = min_coords(
				aView.dimensions()-coord_t::fill(1),
				max_coords(coord_t(),
					aCorner + index_from_linear_access_index(TStaticSize::vector(), index)));
			buffer[index] = aView[coords];

			index += TThreadBlockSize::count();
		}
	}

	CUGIP_DECL_DEVICE
	TElement &
	get(coord_t aCoords)
	{
		TElement *buffer = reinterpret_cast<TElement *>(data);
		return buffer[get_linear_access_index(TStaticSize::vector(), aCoords)];
	}
protected:

	int8_t data[cBufferSize];
};

} // namespace detail
} // namespace cugip
