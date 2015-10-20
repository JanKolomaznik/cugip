#pragma once

#include <cugip/utils.hpp>
#include <cugip/math.hpp>

namespace cugip {

template <typename TType, int tDimension>
class symmetric_tensor: public simple_vector<TType, ((tDimension + 1) * tDimension) / 2>
{
public:
	typedef simple_vector<TType, ((tDimension + 1) * tDimension) / 2> base_type;

	CUGIP_DECL_HYBRID
	symmetric_tensor()
		: base_type()
	{}

	CUGIP_DECL_HYBRID
	TType &
	get(int aRow, int aCol)
	{
		if (aRow > aCol) {
			swap(aRow, aCol);
		}
		const int index = aCol + aRow * tDimension - ((aRow +1) * aRow / 2);
		return (*this)[index];
	}

	CUGIP_DECL_HYBRID
	const TType &
	get(int aRow, int aCol) const
	{
		if (aRow > aCol) {
			swap(aRow, aCol);
		}
		const int index = aCol + aRow * tDimension - ((aRow +1) * aRow / 2);
		return (*this)[index];
	}

	//TODO - get wrapper which provides access
	CUGIP_DECL_HYBRID
	simple_vector<TType, tDimension>
	column(int aCol) const
	{
		simple_vector<TType, tDimension> result;
		for (int i = 0; i < tDimension; ++i) {
			result[i] = get(i, aCol);
		}
		return result;
	}

	CUGIP_DECL_HYBRID
	simple_vector<TType, tDimension>
	row(int aRow) const
	{
		return column(aRow);
	}

private:

};

template <typename TType, int tDimension>
TType
matrix_trace(const symmetric_tensor<TType, tDimension> &aMatrix)
{
	TType result = 0;
	for (int i = 0; i < tDimension; ++i) {
		result += aMatrix.get(i, i);
	}
	return result;
}

template<int tIdx1, int tIdx2, typename TType, int tDimension>
struct get_policy2<tIdx1, tIdx2, const symmetric_tensor<TType, tDimension>>
{
	typedef const TType & return_type;
	typedef const symmetric_tensor<TType, tDimension> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		if (tIdx1 <= tIdx2) {
			constexpr int index = tIdx2 + tIdx1 * tDimension - ((tIdx1 +1) * tIdx1 / 2);
			//static_assert(index < dimension(aArg), "Index overflow");
			return aArg[index];
		} else {
			constexpr int index = tIdx1 + tIdx2 * tDimension - ((tIdx2 +1) * tIdx2 / 2);
			//static_assert(index < dimension(aArg), "Index overflow");
			return aArg[index];
		}
	}
};

template<int tIdx1, int tIdx2, typename TType, int tDimension>
struct get_policy2<tIdx1, tIdx2, symmetric_tensor<TType, tDimension>>
{
	typedef TType & return_type;
	typedef symmetric_tensor<TType, tDimension> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		if (tIdx1 <= tIdx2) {
			constexpr int index = tIdx2 + tIdx1 * tDimension - ((tIdx1 +1) * tIdx1 / 2);
			//static_assert(index < dimension(aArg), "Index overflow");
			return aArg[index];
		} else {
			constexpr int index = tIdx1 + tIdx2 * tDimension - ((tIdx2 +1) * tIdx2 / 2);
			//static_assert(index < dimension(aArg), "Index overflow");
			return aArg[index];
		}
	}
};


}  // namespace cugip
