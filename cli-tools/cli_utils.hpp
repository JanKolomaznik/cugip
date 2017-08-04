#pragma once

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <boost/spirit/home/x3.hpp>
#include <boost/format.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>

namespace cmd {
struct Range {
	float from;
	float to;
};

void validate(boost::any& v, const std::vector<std::string>& values, Range* target_type, int)
{
	using namespace boost::program_options;
	namespace ascii = boost::spirit::x3::ascii;
	namespace x3 = boost::spirit::x3;

	// Make sure no previous assignment to 'a' was made.
	validators::check_first_occurrence(v);
	// Extract the first string from 'values'. If there is more than
	// one string, it's an error, and exception will be thrown.
	const std::string& s = validators::get_single_string(values);

	using x3::int_;
	using x3::lit;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::tuple<float, float> result;
	auto rule = lit('[') >> float_ >> lit(',') >> float_  >> lit(']');
	bool const res = x3::phrase_parse(s.begin(), s.end(), rule, blank, result);
	if (res) {
		v = boost::any(Range{ std::get<0>(result), std::get<1>(result) });
	} else {
		throw validation_error(validation_error::invalid_option_value);
	}
}

} // namespace cmd
