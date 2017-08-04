#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "cli_utils.hpp"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <cugip/host_image_view.hpp>


namespace po = boost::program_options;
namespace fs = boost::filesystem;


int main( int argc, char* argv[] )
{
	fs::path inputFile;
	fs::path outputFile;

	cmd::Range scaleRange;
	int levelCount
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<fs::path>(&inputFile), "input file")
		("output,o", po::value<fs::path>(&outputFile), "output file")
		("scale,s", po::value<cmd::Range>(&scaleRange)/*->default_value(cmd::Range{ 1.0f, 1.0f })*/, "scale range")
		("levels,l", po::value<int>(&levelCount)->default_value(1), "level count")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}

	if (vm.count("input") == 0) {
		std::cout << "Missing input filename\n" << desc << "\n";
		return 1;
	}

	if (vm.count("output") == 0) {
		std::cout << "Missing output filename\n" << desc << "\n";
		return 1;
	}

	typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

	try {
		/*itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(inputFile.c_str(), itk::ImageIOFactory::ReadMode);
		imageIO->SetFileName(inputFile.string());
		imageIO->ReadImageInformation();
		const ScalarPixelType pixelType = imageIO->GetComponentType();
		std::cout << "Pixel Type is " << imageIO->GetComponentTypeAsString(pixelType) << std::endl;

		const size_t numDimensions =  imageIO->GetNumberOfDimensions();
		std::cout << "numDimensions: " << numDimensions << std::endl;

		if (imageIO->GetNumberOfComponents () != 1) {
			std::cerr << "Error: Only scalar images supported.\n";
			return EXIT_FAILURE;
		}

		switch (numDimensions) {
		case 2:
			processImage<2, int64_t>(inputFile, outputFile, options);
			break;
		case 3:
			processImage<3, int64_t>(inputFile, outputFile, options);
			break;
		default:
			std::cerr << "Error: Unsupported image dimension.\n";
			return EXIT_FAILURE;
		}*/

	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
