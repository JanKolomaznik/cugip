
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>

using namespace boost::gil;

void
process(rgb8_image_t::const_view_t aIn, rgb8_image_t::view_t aOut); 


template <typename Out>
struct halfdiff_cast_channels {
    template <typename T> Out operator()(const T& in1, const T& in2) const {
        return Out((in2-in1)/2);
    }
};


template <typename SrcView, typename DstView>
void x_gradient(const SrcView& src, const DstView& dst) {
    typedef typename channel_type<DstView>::type dst_channel_t;

    for (int y=0; y<src.height(); ++y) {
        typename SrcView::x_iterator src_it = src.row_begin(y);
        typename DstView::x_iterator dst_it = dst.row_begin(y);

        for (int x=1; x<src.width()-1; ++x) {
            static_transform(src_it[x-1], src_it[x+1], dst_it[x], 
                             halfdiff_cast_channels<dst_channel_t>());
        }
    }
}

template <typename SrcView, typename DstView>
void x_luminosity_gradient(const SrcView& src, const DstView& dst) {
    typedef pixel<typename channel_type<SrcView>::type, gray_layout_t> gray_pixel_t;
    x_gradient(color_converted_view<gray_pixel_t>(src), dst);
}


int main() {
    rgb8_image_t img;
    jpeg_read_image("test.jpg",img);
    
    rgb8_image_t img_out(img.dimensions());
    //fill_pixels(view(img_out),bits8s(0));

    process(const_view(img), view(img_out));

    //x_luminosity_gradient(const_view(img), view(img_out));
    jpeg_write_view("out-x_gradient.jpg",const_view(img_out));

    return 0;
}
