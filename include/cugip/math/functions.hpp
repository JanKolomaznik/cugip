
namespace cugip {



#define EPSILON 0.000001f

template<typename TType>
inline CUGIP_DECL_HYBRID TType
sqr(TType aValue) {
	return aValue * aValue;
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
abs(TType aValue) {
	if (aValue < 0) {
		return -1 * aValue;
	}
	return aValue;
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
max(TType aValue1, TType aValue2) {
	return aValue1 < aValue2 ? aValue2 : aValue1;
}

template<typename TType>
inline CUGIP_DECL_HYBRID TType
min(TType aValue1, TType aValue2) {
	return aValue1 < aValue2 ? aValue1 : aValue2;
}



}//namespace cugip
