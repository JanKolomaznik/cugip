#pragma once

namespace cugip {
namespace detail {

template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
void
transformHost(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		aAssignOperation(linear_access(aOutView, i), aOperator(linear_access(aInView, i)));
	}
}

template <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
void
transformHost(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView1); ++i) {
		aAssignOperation(linear_access(aOutView, i), aOperator(linear_access(aInView1, i), linear_access(aInView2, i)));
	}
}

template<>
struct TransformImplementation<false> {
	template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::transformHost(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
	}

	template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
	static void run(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::transformHost(aInView1, aInView1, aOutView, aOperator, aAssignOperation, aPolicy);
	}
};

template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
void
transformPositionHost(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		aAssignOperation(linear_access(aOutView, i), aOperator(linear_access(aInView, i)));
	}
}

template<>
struct TransformPositionImplementation<false> {
	template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::transformPositionHost(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
	}
};

template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
void
transformLocatorHost(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		aAssignOperation(linear_access(aOutView, i), aOperator(create_locator<TInView, typename TPolicy::BorderHandling>(aInView, index_from_linear_access_index(aInView, i))));
	}
}

template<>
struct TransformLocatorImplementation<false> {

	template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::transformLocatorHost(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
	}
};


} // namespace detail
} // namespace cugip
