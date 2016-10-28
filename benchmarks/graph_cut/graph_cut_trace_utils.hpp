#pragma once


struct CudacutConfig {
	CudacutConfig(
		cugip::host_image_view<uint8_t, 3> aSaturated,
		cugip::host_image_view<uint8_t, 3> aExcess,
		cugip::host_image_view<float, 3> aLabels,
		float aResidualThreshold)
			: saturated(aSaturated)
			, excess(aExcess)
			, labels(aLabels)
			, residualThreshold(aResidualThreshold)
	{}
	cugip::host_image_view<uint8_t, 3> saturated;
	cugip::host_image_view<uint8_t, 3> excess;
	cugip::host_image_view<float, 3> labels;
	float residualThreshold = 0.0f;
};
