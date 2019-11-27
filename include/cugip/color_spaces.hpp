#pragma once

namespace cugip {

class RGB_8: public simple_vector<uint8_t, 3> {};
class RGB_16: public simple_vector<uint16_t, 3> {};
class RGB_32: public simple_vector<uint32_t, 3> {};
class RGB_32f: public simple_vector<float, 3> {};
class RGB_64f: public simple_vector<double, 3> {};

class RGBA_8: public simple_vector<uint8_t, 4> {};
class RGBA_16: public simple_vector<uint16_t, 4> {};
class RGBA_32: public simple_vector<uint32_t, 4> {};
class RGBA_32f: public simple_vector<float, 4> {};
class RGBA_64f: public simple_vector<double, 4> {};

} // namespace cugip
