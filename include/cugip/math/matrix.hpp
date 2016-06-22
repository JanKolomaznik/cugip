#pragma once

#include <cugip/utils.hpp>
#include <cugip/math.hpp>

namespace cugip {

template<typename TMatrix>
class column_accessor
{
public:
	CUGIP_DECL_HYBRID
	auto get(int aIdx) const -> decltype(std::declval<TMatrix>().get(aIdx, 0))
	{
		return mMatrix.get(aIdx, mColumn);
	}

	CUGIP_DECL_HYBRID
	auto operator[](int aIdx) const  -> decltype(std::declval<TMatrix>().get(aIdx, 0))
	{
		return mMatrix.get(aIdx, mColumn);
	}

	template<typename TVector>
	CUGIP_DECL_HYBRID typename std::enable_if<static_vector_traits<TVector>::is_vector, column_accessor<TMatrix> &>::type
	operator=(const TVector &aVector)
	{
		static_assert(static_vector_traits<TVector>::dimension == TMatrix::cRowCount, "Vectors must have same dimension to be assignable");
		for (int i = 0; i < TMatrix::cRowCount; ++i) {
			mMatrix.get(i, mColumn) = aVector[i];
		}
		return *this;
	}

	int mColumn;
	TMatrix &mMatrix;
};

template<typename TMatrix>
struct static_vector_traits<column_accessor<TMatrix>>
{
	static constexpr bool is_vector = true;
	static constexpr int dimension = TMatrix::cRowCount;

	typedef typename TMatrix::element_t element_type;
};


template <typename TType, int tRowCount, int tColCount>
class matrix: public simple_vector<simple_vector<TType, tColCount>, tRowCount>
{
public:
	typedef simple_vector<simple_vector<TType, tColCount>, tRowCount> base_type;
	static constexpr int cRowCount = tRowCount;
	static constexpr int cColCount = tColCount;

	typedef TType element_t;

	CUGIP_DECL_HYBRID
	matrix()
		: base_type()
	{}

	CUGIP_DECL_HYBRID
	TType &
	get(int aRow, int aCol)
	{
		return (*this)[aRow][aCol];
	}

	CUGIP_DECL_HYBRID
	const TType &
	get(int aRow, int aCol) const
	{
		return (*this)[aRow][aCol];
	}

	//TODO - get wrapper which provides access
	CUGIP_DECL_HYBRID
	simple_vector<TType, tRowCount>
	column(int aCol) const
	{
		simple_vector<TType, tRowCount> result;
		for (int i = 0; i < tRowCount; ++i) {
			result[i] = get(i, aCol);
		}
		return result;
	}

	CUGIP_DECL_HYBRID
	column_accessor<matrix<TType, tRowCount, tColCount>>
	column(int aCol)
	{
		return column_accessor<matrix<TType, tRowCount, tColCount>>{aCol, *this};
	}

	CUGIP_DECL_HYBRID
	simple_vector<TType, tColCount>
	row(int aRow) const
	{
		return (*this)[aRow];
	}

	CUGIP_DECL_HYBRID
	simple_vector<TType, tColCount> &
	row(int aRow)
	{
		return (*this)[aRow];
	}

private:

};

template <typename TType, int tRowCount>
CUGIP_DECL_HYBRID TType
matrix_trace(const matrix<TType, tRowCount, tRowCount> &aMatrix)
{
	TType result = 0;
	for (int i = 0; i < tRowCount; ++i) {
		result += aMatrix.get(i, i);
	}
	return result;
}

template <int tIdx1, int tIdx2, typename TType, int tRowCount, int tColCount>
struct get_policy2<tIdx1, tIdx2, const matrix<TType, tRowCount, tColCount>>
{
	typedef const TType & return_type;
	typedef const matrix<TType, tRowCount, tColCount> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		return get<tIdx2>(get<tIdx1>(aArg));
	}
};

template <int tIdx1, int tIdx2, typename TType, int tRowCount, int tColCount>
struct get_policy2<tIdx1, tIdx2, matrix<TType, tRowCount, tColCount>>
{
	typedef TType & return_type;
	typedef matrix<TType, tRowCount, tColCount> & value_t;

	static CUGIP_DECL_HYBRID return_type
	get(value_t aArg)
	{
		return get<tIdx2>(get<tIdx1>(aArg));
	}
};


}  // namespace cugip

