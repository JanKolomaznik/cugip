#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

typedef itk::Image<float, 3> ImageType;
typedef itk::Image<int, 3> MarkersType;

int
main(int argc, char* argv[])
{
	std::string inputFile;
	std::string markersFile;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&inputFile), "input file")
		("markers,m", po::value<std::string>(&markersFile), "markers file")
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

	if (vm.count("markers") == 0) {
		std::cout << "Missing markers filename\n" << desc << "\n";
		return 1;
	}

	typedef itk::ImageFileReader<ImageType>  ImageReaderType;
	typedef itk::ImageFileReader<MarkersType>  MarkersReaderType;

	ImageReaderType::Pointer imageReader = ImageReaderType::New();
	MarkersReaderType::Pointer markersReader = MarkersReaderType::New();
	imageReader->SetFileName(inputFile);
	markersReader->SetFileName(markersFile);
	imageReader->Update();
	markersReader->Update();

	ImageType::Pointer image = imageReader->GetOutput();
	MarkersType::Pointer markers = markersReader->GetOutput();

	/*ImageType::Pointer output_image = ImageType::New();
	output_image->SetRegions(image->GetLargestPossibleRegion());
	output_image->Allocate();

	//denoise(image, output_image);
	denoise(
		image->GetPixelContainer()->GetBufferPointer(),
		output_image->GetPixelContainer()->GetBufferPointer(),
		image->GetLargestPossibleRegion().GetSize()[0],
		image->GetLargestPossibleRegion().GetSize()[1],
		image->GetLargestPossibleRegion().GetSize()[2]);


	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(output_file);
	writer->SetInput(output_image);

	try
	{
		writer->Update();
	}
	catch( itk::ExceptionObject & error )
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}*/

	return EXIT_SUCCESS;
}
