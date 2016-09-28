
namespace cugip {
namespace detail {


template <typename TView, typename TFunctor, typename TPolicy>
CUGIP_GLOBAL void
kernel_for_each(TView aView, TFunctor aOperator, TPolicy aPolicy)
{
	typename TView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView>::value>();
	typename TView::extents_t extents = aView.dimensions();

	if (coord < extents) {
		aView[coord] = aOperator(aView[coord]);
	}
}

template<>
struct ForEachImplementation<true> {

	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView);

		detail::kernel_for_each<TInView, TFunctor, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOperator, aPolicy);
	}
};


template <typename TView, typename TFunctor, typename TPolicy>
CUGIP_GLOBAL void
kernel_for_each_position(TView aView, TFunctor aOperator, TPolicy aPolicy)
{
	typename TView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView>::value>();
	typename TView::extents_t extents = aView.dimensions();

	if (coord < extents) {
		aView[coord] = aOperator(aView[coord], coord);
	}
}

template<>
struct ForEachPositionImplementation<true> {

	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView);

		detail::kernel_for_each_position<TInView, TFunctor, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOperator, aPolicy);
	}
};

}//namespace detail
}//namespace cugip
