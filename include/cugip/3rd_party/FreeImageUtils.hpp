#pragma once

#include <daimas/common/definitions.hpp>

#include <FreeImage.h>
#include <boost/exception/all.hpp>
#include <boost/format.hpp>
#include <boost/exception/all.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/dynamic_image/dynamic_image_all.hpp>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <iterator>



struct EFileNotExists: virtual std::exception, virtual boost::exception {};
struct EUnsupportedImageFormat: virtual std::exception, virtual boost::exception {};
struct EUnsupportedColorSpace: virtual std::exception, virtual boost::exception {};
struct EImageReadFailure: virtual std::exception, virtual boost::exception {};
struct EImageWriteFailure: virtual std::exception, virtual boost::exception {};
typedef boost::error_info<struct tag_path, fs::path> ePathInfo;
typedef boost::error_info<struct tag_description, std::string> eDescription;

//boost::mpl::insert_range<v,pos,r>::type

typedef boost::mpl::vector< 
		gil::gray8_image_t,
		gil::gray16_image_t,
		gil::gray32f_image_t
		> SupportedGrayScaleImages;

typedef boost::mpl::vector< 
		gil::rgb8_image_t,
		gil::rgb16_image_t,
		gil::rgb32f_image_t
		> SupportedRGBImages;

/*typedef boost::mpl::insert< 
		SupportedGrayScaleImages::type,
		boost::mpl::size<SupportedGrayScaleImages>::type,
		SupportedRGBImages
		> SupportedImageTypes;*/

typedef boost::mpl::vector< 
		gil::gray8_image_t,
		gil::gray16_image_t,
		gil::gray32f_image_t,
		gil::rgb8_image_t,
		gil::rgb16_image_t,
		gil::rgb32f_image_t
		> SupportedImageTypes;

typedef gil::any_image< SupportedImageTypes > AnyImage;
typedef gil::any_image_view< SupportedImageTypes > AnyImageView;
typedef boost::shared_ptr< AnyImage > AnyImagePtr;
typedef gil::rgb32f_image_t ImageRGBf;
typedef ImageRGBf::view_t ViewRGBf;
typedef ImageRGBf::const_view_t ConstViewRGBf;


/*template <typename TView>
struct AnyImageGetView {
	typedef TView result_type;
	template <typename TImage> result_type operator()(TImage& img) const { return result_type(gil::view(img)); }
};*/

template <typename TTargetImage>
struct AnyImageColorConversion {
	typedef void result_type;
	AnyImageColorConversion(TTargetImage &aTarget) : target(aTarget) {}

	template <typename TImage> result_type operator()(TImage& img) {
		gil::copy_pixels(gil::color_converted_view<typename TTargetImage::value_type>(gil::const_view(img)), gil::view(target));
	}
	TTargetImage &target;
};

struct FreeImageInitializer
{
	FreeImageInitializer();
	
	~FreeImageInitializer()
	{
		FreeImage_DeInitialise();
	}
};

//-----------------------------------------------------------------------------

class ImageBucket
{
public:
	typedef boost::shared_ptr< ImageBucket > Ptr;

	ImageBucket(AnyImagePtr aImageData = AnyImagePtr()): mContents(aImageData)
	{}

	AnyImage &
	imageData() {
		assert(mContents);
		return *mContents;
	}

	AnyImagePtr mContents;
};



ImageBucket::Ptr
readImage(fs::path aFileName);

AnyImagePtr
readImageData(fs::path aFileName);


template <typename TPaths, typename TImages>
void
loadImages(typename boost::call_traits<TPaths>::const_reference aPaths,
	   typename boost::call_traits<TImages>::reference aImages)
{
	std::for_each(std::begin(aPaths), std::end(aPaths), [&](const fs::path &aPath) { aImages.push_back(readImage(aPath)); });
}

template<typename TView>
FIBITMAP* 
getFiBitmapFromView( const TView &aView )
{
	auto locator = aView.xy_at(0,0);
	
			std::cout << "Bitmap RGB " << locator.row_size() << "\n";
			std::cout << "\t" << (size_t)(&(locator(0,1))) << ";" << (size_t)( &(locator(0,0))) << "\n";
			std::cout << "\t" << (size_t)((char*)&(locator(0,1)) - (char*)&(locator(0,0))) << "\n";
			std::cout << "\t" << sizeof(gil::rgb8_image_t::value_type) << "\n";
	return FreeImage_ConvertFromRawBits((BYTE*)&locator(0,0), aView.width(), aView.height(), locator.row_size(), 3*8, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
}

template<typename TView>
void
writeImageData( fs::path aFileName, const TView &aView )
{
	assert(aView.width() && aView.height());

	FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(aFileName.c_str());
	FIBITMAP *bitmap = getFiBitmapFromView(aView);
	if(bitmap)
	{
		try {
			if(fif == FIF_UNKNOWN || !FreeImage_FIFSupportsWriting(fif) || fif==FIF_GIF) {
				BOOST_THROW_EXCEPTION( EUnsupportedImageFormat() );
			}

			int flags = 0;
			if(fif==FIF_JPEG) {
				flags = 95; //compression
			}

			if ( !FreeImage_Save(fif, bitmap, aFileName.c_str(), flags) ) {
				BOOST_THROW_EXCEPTION( EImageWriteFailure() );
			}
			FreeImage_Unload(bitmap);
			bitmap = NULL;
		} catch(...) {
			if(bitmap) {
				FreeImage_Unload(bitmap);
			}
			throw;
		}
	}
}
