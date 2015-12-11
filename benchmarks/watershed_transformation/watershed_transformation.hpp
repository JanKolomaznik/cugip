#pragma once


#include <cugip/host_image_view.hpp>

struct Options
{

};

void
watershedTransformation(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::host_image_view<int, 3> aLabels,
	Options aOptions);

void
watershedTransformation2(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::host_image_view<int, 3> aLabels,
	Options aOptions);
