#define BOOST_TEST_MODULE GeometryTransformationTest
//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cmath>

#include <cuda.h>

#include <cugip/utils.hpp>
#include <cugip/reduce.hpp>
#include <cugip/math.hpp>
#include <cugip/image.hpp>
#include <cugip/host_image.hpp>
#include <cugip/copy.hpp>
#include <cugip/procedural_views.hpp>
#include <cugip/for_each.hpp>
#include <cugip/interpolated_view.hpp>
#include <cugip/geometry_transformation.hpp>

using namespace cugip;

static const float cEpsilon = 0.00001;


BOOST_AUTO_TEST_CASE(RotationTest)
{
	host_image<float, 2> output(20, 20);

	auto inputView = cugip::checkerBoard<float, 2>(0.0f, 1.0f, Int2(10, 10), Int2(20, 20));
	auto testView = cugip::checkerBoard<float, 2>(1.0f, 0.0f, Int2(10, 10), Int2(20, 20));
	rotate(make_interpolated_view(inputView), view(output), vect2f_t(10.0f, 10.0f), float(M_PI / 2.0f));

	auto difference = sum_differences(view(output), testView, 0.0f);
	BOOST_CHECK_CLOSE(difference, 0.0f, cEpsilon);
}

BOOST_AUTO_TEST_CASE(ScaleTest)
{
	host_image<float, 2> output(40, 40);

	auto inputView = cugip::checkerBoard<float, 2>(0.0f, 1.0f, Int2(10, 10), Int2(20, 20));
	auto testView = cugip::checkerBoard<float, 2>(0.0f, 1.0f, Int2(20, 20), Int2(40, 40));
	scale(make_interpolated_view(inputView), view(output), vect2f_t(0.0f, 0.0f), vect2f_t(2.0f, 2.0f));

	auto difference = sum_differences(view(output), testView, 0.0f);
	BOOST_CHECK_CLOSE(difference, 0.0f, cEpsilon);
}

BOOST_AUTO_TEST_CASE(LinearInterpolationScaleTest)
{
	host_image<float, 2> output(40, 40);

	auto inputView = cugip::checkerBoard<float, 2>(0.0f, 1.0f, Int2(10, 10), Int2(20, 20));
	auto testView = cugip::checkerBoard<float, 2>(0.0f, 1.0f, Int2(20, 20), Int2(40, 40));
	scale(make_interpolated_view(inputView, LinearInterpolator<BorderHandlingTraits<border_handling_enum::REPEAT>>()), view(output), vect2f_t(0.0f, 0.0f), vect2f_t(2.0f, 2.0f));

	dump_view_to_file(inputView, "scale_input_20x20.raw");
	dump_view_to_file(view(output), "scale_test_40x40.raw");
	auto difference = sum_differences(view(output), testView, 0.0f);
	BOOST_CHECK_CLOSE(difference, 0.0f, cEpsilon);
}
