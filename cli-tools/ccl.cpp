#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <cugip/host_image_view.hpp>

#include "watershed_options.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace cugip;

void runConnectedComponentLabeling(
		const_host_image_view<const int8_t, 2> aInput,
		host_image_view<int32_t, 2> aOutput,
		const std::string &aName);

void runConnectedComponentLabeling(
		const_host_image_view<const int8_t, 3> aInput,
		host_image_view<int32_t, 3> aOutput,
		const std::string &aName);


template<int tDimension>
void processImage(fs::path aInput, fs::path aOutput, std::string aMethodName)
{
	typedef itk::Image<int8_t, tDimension> InputImageType;
	typedef itk::Image<int32_t, tDimension> OutputImageType;

	typedef itk::ImageFileReader<InputImageType>  ReaderType;
	typedef itk::ImageFileWriter<OutputImageType> WriterType;

	typename ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(aInput.string());
	reader->Update();

	typename InputImageType::Pointer image = reader->GetOutput();

	typename OutputImageType::Pointer output_image = OutputImageType::New();
	output_image->SetRegions(image->GetLargestPossibleRegion());
	output_image->Allocate();
	output_image->SetSpacing(image->GetSpacing());

	cugip::simple_vector<int, tDimension> size;
	for (int i = 0; i < tDimension; ++i) {
		size[i] = image->GetLargestPossibleRegion().GetSize()[i];
	}

	auto inView = makeConstHostImageView(image->GetPixelContainer()->GetBufferPointer(), size);
	auto outView = makeHostImageView(output_image->GetPixelContainer()->GetBufferPointer(), size);

	runConnectedComponentLabeling(inView, outView, aMethodName);

	typename WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(aOutput.string());
	writer->SetInput(output_image);
	writer->Update();
}

int main( int argc, char* argv[] )
{
	fs::path inputFile;
	fs::path outputFile;

	WatershedOptions options;
	std::string methodName;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<fs::path>(&inputFile), "input file")
		("output,o", po::value<fs::path>(&outputFile), "output file")
		("type,t", po::value<std::string>(&methodName)->default_value("async_gs"), "async_gs, sync_gs")
		//("normalize,n", po::value<bool>(&normalize)->default_value(false), "normalize")
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
		itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(inputFile.c_str(), itk::ImageIOFactory::ReadMode);
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
			processImage<2>(inputFile, outputFile, methodName);
			break;
		case 3:
			processImage<3>(inputFile, outputFile, methodName);
			break;
		default:
			std::cerr << "Error: Unsupported image dimension.\n";
			return EXIT_FAILURE;
		}

	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
