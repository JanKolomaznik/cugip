
namespace cugip {
namespace detail {


template <typename TInView, typename TFunctor, typename TPolicy>
void
for_each_host(TInView aInView, TFunctor aOperator, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		aOperator(linear_access(aInView, i));
	}
}

template<>
struct ForEachImplementation<false> {
	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::for_each_host(aInView, aOperator, aPolicy);
	}
};


template <typename TInView, typename TFunctor, typename TPolicy>
void
for_each_position_host(TInView aInView, TFunctor aOperator, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		aOperator(linear_access(aInView, i));
	}
}

template<>
struct ForEachPositionImplementation<false> {
	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::for_each_position_host(aInView, aOperator, aPolicy);
	}
};


}//namespace detail
}//namespace cugip
