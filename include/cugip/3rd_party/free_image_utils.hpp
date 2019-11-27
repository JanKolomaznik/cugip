#pragma once

#include <FreeImage.h>

#include <boost/filesystem.hpp>
#include <boost/variant.hpp>

#include <cugip/exception.hpp>
#include <cugip/host_image.hpp>
#include <cugip/any_image.hpp>
#include <cugip/color_spaces.hpp>

namespace cugip {

struct EFileNotExists: public ExceptionBase {};
struct EUnsupportedImageFormat: public ExceptionBase {};
struct EUnsupportedColorSpace: public ExceptionBase {};
struct EImageReadFailure: public ExceptionBase {};
struct EImageWriteFailure: public ExceptionBase {};

typedef boost::error_info<struct tag_path, boost::filesystem::path> PathErrorInfo;
typedef boost::error_info<struct tag_description, std::string> DescriptionErrorInfo;


struct FreeImageInitializer
{
	FreeImageInitializer()
	{
		FreeImage_Initialise();
	}

	~FreeImageInitializer()
	{
		FreeImage_DeInitialise();
	}
};

struct ImageDescription {
	ImageDescription(FIBITMAP &aBitmap)
	{
		imageType = FreeImage_GetImageType(&aBitmap);
		width = FreeImage_GetWidth(&aBitmap);
		height = FreeImage_GetHeight(&aBitmap);

		colorType = FreeImage_GetColorType(&aBitmap);
	}

	FREE_IMAGE_TYPE imageType;
	unsigned width;
	unsigned height;
	FREE_IMAGE_COLOR_TYPE colorType;
};

AnyImage
convertFreeImageToCUGIP(FIBITMAP &aBitmap)
{
	auto description = ImageDescription(aBitmap);

	switch (description.colorType) {
	case FIC_MINISBLACK: {
		/*switch (imageType) {
		case FIT_BITMAP:
			std::cout << "Bitmap \n";
			break;
		case FIT_UINT16:{
			std::cout << "FIT_UINT16 \n";
			//gil::gray16_image_t image(width, height);
			break;
		}
		case FIT_FLOAT:
			std::cout << "FIT_FLOAT \n";
		break;
		case FIT_INT16:
		case FIT_UINT32:
		case FIT_INT32:
		case FIT_DOUBLE:
		case FIT_COMPLEX:
		default:
			CUGIP_THROW( EUnsupportedImageFormat() );
		}*/
		CUGIP_THROW(EUnsupportedImageFormat() << DescriptionErrorInfo("Unsupported single channel images!"));
		break;
	}
	case FIC_RGB: {
		switch (description.imageType) {
		case FIT_BITMAP: {
			host_image<RGB_8, 2> targetImage(description.width, description.height);
			FreeImage_ConvertToRawBits(
					reinterpret_cast<BYTE *>(targetImage.pointer()),
					&aBitmap,
					targetImage.strides()[1],
					24,
					FI_RGBA_RED_MASK,
					FI_RGBA_GREEN_MASK,
					FI_RGBA_BLUE_MASK,
					TRUE);
			return AnyImage(std::move(targetImage));
		}
		case FIT_RGB16:
		case FIT_RGBF:
		case FIT_RGBA16:
		case FIT_RGBAF:
		default:
			CUGIP_THROW( EUnsupportedImageFormat() << DescriptionErrorInfo("Unsupported color space"/*gMapFIImageType[imageType]*/));
		}
		break;
	}
	default:
		BOOST_THROW_EXCEPTION( EUnsupportedColorSpace() );
	}
	return AnyImage();
}



FREE_IMAGE_FORMAT
getImageFormat(boost::filesystem::path aFileName)
{
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	if (!boost::filesystem::exists(aFileName)) {
		CUGIP_THROW( EFileNotExists() << PathErrorInfo(aFileName) );
	}

	fif = FreeImage_GetFileType(aFileName.c_str(), 0);
	if (fif == FIF_UNKNOWN){
		fif = FreeImage_GetFIFFromFilename(aFileName.c_str());

	}
	if ((fif == FIF_UNKNOWN)
		|| !FreeImage_FIFSupportsReading(fif)
		//|| !isSupportedImageFormat(fif)
	)
	{
		CUGIP_THROW( EUnsupportedImageFormat() << PathErrorInfo(aFileName) );
	}
	return fif;
}

AnyImage
readImageData(boost::filesystem::path aFileName)
{
	FREE_IMAGE_FORMAT fif = getImageFormat(aFileName);

	int flags = 0;
	if(fif==FIF_JPEG) {
		flags = JPEG_ACCURATE;
	}
	std::unique_ptr<FIBITMAP, decltype(&FreeImage_Unload)> bitmap(FreeImage_Load(fif, aFileName.c_str(), flags), &FreeImage_Unload);
	if(not bitmap) {
		CUGIP_THROW(EImageReadFailure() << PathErrorInfo(aFileName));
	}
	return convertFreeImageToCUGIP(*bitmap);
}


/*template<typename TView>
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
				CUGIP_THROW( EUnsupportedImageFormat() );
			}

			int flags = 0;
			if(fif==FIF_JPEG) {
				flags = 95; //compression
			}

			if ( !FreeImage_Save(fif, bitmap, aFileName.c_str(), flags) ) {
				CUGIP_THROW( EImageWriteFailure() );
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
}*/

} // namespace cugip
