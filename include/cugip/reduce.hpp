#pragma once

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <type_traits>
#include <cugip/image_view.hpp>
#include <cugip/access_utils.hpp>

namespace cugip {

struct ReduceExecutionConfig {
	cudaStream_t stream = 0;
};



/** \addtogroup meta_algorithm
 * @{
 **/

/// Implements parallel reduction - application of associative operator on all image elements.
/// For example to sum all values in integer image:
/// \code
/// 	sum = reduce(view, 0, thrust::plus<int>());
/// \endcode
/// \param view Processed image view - only constant element access needed.
/// \param initial_value Value used for result initialization (0 for sums, 1 for products, etc.)
/// \param reduction_operator Associative operator - it must be callable on device like this: result = reduction_operator(val1, val2).
/// \return Result of the operation.
template<typename TView, typename TOutputValue, typename TOperator>
TOutputValue reduce(TView view, TOutputValue initial_value, TOperator reduction_operator);

template<typename TView, class = typename std::enable_if<is_image_view<TView>::value>::type>
typename TView::value_type min(TView view);

template<typename TView, class = typename std::enable_if<is_image_view<TView>::value>::type>
typename TView::value_type max(TView view);

template<typename TView, typename TOutputValue, class = typename std::enable_if<is_image_view<TView>::value>::type>
TOutputValue sum(TView view, TOutputValue initial_value);

/// WARNING: as a result type is used element type of the passed view, so be aware of possible overflow if summing large view of small type elements (int8_t, int16_t, ...)
/// When in doubt use Sum(view, initial_value); and pass initial value of bigger type to prevent overflows.
template<typename TView, class = typename std::enable_if<is_image_view<TView>::value>::type>
typename TView::value_type sum(TView view);

template<typename TView1, typename TView2, typename TOutputValue, class = typename std::enable_if<is_image_view<TView1>::value && is_image_view<TView2>::value>::type>
TOutputValue sum_differences(TView1 view1, TView2 view2, TOutputValue initial_value);

/**
 * @}
 **/

}  // namespace cugip

#include <cugip/reduce.tcc>
