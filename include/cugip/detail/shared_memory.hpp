#prama once

#include <static_int_sequence.hpp>

namespace cugip {


namespace detail {

template<typename TElement, typename TStaticSize, typename TThreadBlockSize>
class SharedMemory
{
public:
	static constexpr int cDimension = TStaticSize::cDimension;
	static constexpr int cElementCount = TStaticSize::count();
	static constexpr int cBufferSize = cElementCount * sizeof(TElement);

	typedef simple_vector<int, cDimension> coord_t;

	template<typename TView>
	void load(TView aView, region<cDimension> aRegion)
	{
		int index = threadOrderFromIndex();
		TElement *buffer = reinterpret_cast<TElement *>(data);
		while (index < cElementCount) {
			buffer[index] = aView[region_linear_access(aRegion, index)];

			index += TThreadBlockSize::count();
		}
	}
protected:

	int8_t data[cBufferSize];
};

} // namespace detail
} // namespace cugip
