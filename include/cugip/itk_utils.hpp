#pragma once

#include <itkImage.h>
#include <cugip/host_image_view.hpp>

namespace cugip {

template<typename TElement>
host_image_view<TElement, 3>
view(itk::Image<TElement, 3> &aImage)
{
	auto pointer = aImage.GetBufferPointer();
	auto size = Int3(
			aImage.GetLargestPossibleRegion().GetSize()[0],
			aImage.GetLargestPossibleRegion().GetSize()[1],
			aImage.GetLargestPossibleRegion().GetSize()[2]);
	return cugip::makeHostImageView<TElement, 3>(pointer, size);
}

template<typename TElement>
host_image_view<TElement, 2>
view(itk::Image<TElement, 2> &aImage)
{
	auto pointer = aImage.GetBufferPointer();
	auto size = Int2(
			aImage.GetLargestPossibleRegion().GetSize()[0],
			aImage.GetLargestPossibleRegion().GetSize()[1]);
	return cugip::makeHostImageView<TElement, 2>(pointer, size);
}

template<typename TElement>
const_host_image_view<const TElement, 3>
const_view(const itk::Image<TElement, 3> &aImage)
{
	auto pointer = aImage.GetBufferPointer();
	auto size = Int3(
			aImage.GetLargestPossibleRegion().GetSize()[0],
			aImage.GetLargestPossibleRegion().GetSize()[1],
			aImage.GetLargestPossibleRegion().GetSize()[2]);
	return cugip::makeConstHostImageView<TElement, 3>(pointer, size);
}

template<typename TElement>
const_host_image_view<const TElement, 2>
const_view(const itk::Image<TElement, 2> &aImage)
{
	auto pointer = aImage.GetBufferPointer();
	auto size = Int2(
			aImage.GetLargestPossibleRegion().GetSize()[0],
			aImage.GetLargestPossibleRegion().GetSize()[1]);
	return cugip::makeConstHostImageView<TElement, 2>(pointer, size);
}

}  // namespace cugip
