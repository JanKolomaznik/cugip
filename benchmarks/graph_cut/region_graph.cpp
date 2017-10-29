#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/filesystem.hpp>
#include <cugip/itk_utils.hpp>

#include <boost/log/trivial.hpp>
#include <string>

namespace po = boost::program_options;

typedef itk::Image<float, 3> ImageType;
typedef itk::Image<int, 3> IntImageType;

void
regionsNeighborhood(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const int, 3> aRegions);

int
main(int argc, char* argv[])
{
	boost::filesystem::path inputFile;
	boost::filesystem::path regionsFile;
	boost::filesystem::path outputFile;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<boost::filesystem::path>(&inputFile), "input file")
		("regions,r", po::value<boost::filesystem::path>(&regionsFile), "regions file")
		("output,o", po::value<boost::filesystem::path>(&outputFile), "output file")
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

	if (vm.count("regions") == 0) {
		std::cout << "Missing regions filename\n" << desc << "\n";
		return 1;
	}

	if (vm.count("output") == 0) {
		std::cout << "Missing output filename\n" << desc << "\n";
		return 1;
	}

	typedef itk::ImageFileReader<ImageType>  ImageReaderType;
	typedef itk::ImageFileReader<IntImageType>  IntImageReaderType;

	BOOST_LOG_TRIVIAL(info) << "Loading inputs ...";
	ImageReaderType::Pointer imageReader = ImageReaderType::New();
	IntImageReaderType::Pointer regionsReader = IntImageReaderType::New();
	imageReader->SetFileName(inputFile.string());
	regionsReader->SetFileName(regionsFile.string());
	imageReader->Update();
	regionsReader->Update();

	ImageType::Pointer image = imageReader->GetOutput();
	IntImageType::Pointer regions = regionsReader->GetOutput();
	BOOST_LOG_TRIVIAL(info) << "Input size: ["
		<< image->GetLargestPossibleRegion().GetSize()[0] << ", "
		<< image->GetLargestPossibleRegion().GetSize()[1] << ", "
		<< image->GetLargestPossibleRegion().GetSize()[2] << "]";

	BOOST_LOG_TRIVIAL(info) << "Running graph construction ...";
	regionsNeighborhood(
		cugip::const_view(*(image.GetPointer())),
		cugip::const_view(*(regions.GetPointer())));

	/*BOOST_LOG_TRIVIAL(info) << "Saving output `" << outputFile << "` ...";
	MaskWriterType::Pointer writer = MaskWriterType::New();
	writer->SetFileName(outputFile.string());
	writer->SetInput(outputImage);
	try {
		writer->Update();
	} catch (itk::ExceptionObject & error) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}*/

	return EXIT_SUCCESS;
}
