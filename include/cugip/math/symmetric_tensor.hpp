#pragma once

#include <cugip/utils.hpp>
#include <cugip/math.hpp>
#include <cugip/math/matrix.hpp>

namespace cugip {

template <typename TType, int tDimension>
class symmetric_tensor: public simple_vector<TType, ((tDimension + 1) * tDimension) / 2>
{
public:
	static constexpr int cBufferSize = ((tDimension + 1) * tDimension) / 2;
	typedef simple_vector<TType, cBufferSize> base_type;
	static constexpr int cDimension = tDimension;

	CUGIP_DECL_HYBRID
	symmetric_tensor()
		: base_type()
	{}

	CUGIP_DECL_HYBRID
	TType &
	get(int aRow, int aCol)
	{
		if (aRow > aCol) {
			return (*this)[aRow + aCol * tDimension - ((aCol +1) * aCol / 2)];
		}
		return (*this)[aCol + aRow * tDimension - ((aRow +1) * aRow / 2)];
	}

	CUGIP_DECL_HYBRID
	const TType &
	get(int aRow, int aCol) const
	{
		if (aRow > aCol) {
			return (*this)[aRow + aCol * tDimension - ((aCol +1) * aCol / 2)];
		}
		return (*this)[aCol + aRow * tDimension - ((aRow +1) * aRow / 2)];
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
struct static_matrix_traits<symmetric_tensor<TType, tDimension>>
{
	static constexpr bool is_matrix = true;
	static constexpr int row_count = tDimension;
	static constexpr int col_count = tDimension;

	typedef TType element_type;
};

template <typename TType, int tDimension>
CUGIP_DECL_HYBRID TType
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

template<typename TFactor, typename TElement, int tDimension>
inline CUGIP_DECL_HYBRID
	symmetric_tensor<
		decltype(std::declval<TFactor>() * std::declval<TElement>()),
		tDimension>
operator*(const TFactor &aFactor, const symmetric_tensor<TElement, tDimension> &aArg)
{
	symmetric_tensor<
		decltype(std::declval<TFactor>() * std::declval<TElement>()),
		tDimension> res;
	//simple_vector<typename std::common_type<TCoordType1, TCoordType2>::type, tDim> res;
	for (int i = 0; i < symmetric_tensor<TElement, tDimension>::cBufferSize; ++i) {
		res[i] = aFactor * aArg[i];
	}
	return res;
}

template<typename TFactor, typename TElement, int tDimension>
inline CUGIP_DECL_HYBRID
	symmetric_tensor<
		decltype(std::declval<TFactor>() * std::declval<TElement>()),
		tDimension>
operator*(const symmetric_tensor<TElement, tDimension> &aArg, const TFactor &aFactor)
{
	return operator*(aFactor, aArg);
}


}  // namespace cugip
