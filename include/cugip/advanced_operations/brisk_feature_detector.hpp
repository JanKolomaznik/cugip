#pragma once

#include <vector>

namespace cugip {

struct brisk_configuration {

};

struct brisk_point {

};

class brisk_feature_detector {
public:
	brisk_feature_detector() {}

	template<typename TView>
	set_image(TView aView) {

	}

	void allocate() {}

	void run() {}

	const std::vector<brisk_point> &points() const {
		return mPoints;
	}
protected:
	std::vector<brisk_point> mPoints;

};

} // namespace cugip
