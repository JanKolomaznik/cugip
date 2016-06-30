#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

typedef itk::Image<float, 3> ImageType;

//void
//denoise(ImageType::Pointer aInput, ImageType::Pointer aOutput);

void
denoise(float *aInput, float *aOutput, size_t aWidth, size_t aHeight, size_t aDepth, float aVariance);


int main( int argc, char* argv[] )
{
	std::string input_file;
	std::string output_file;
	float variance;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&input_file), "input file")
		("output,o", po::value<std::string>(&output_file), "output file")
		("variance,v", po::value<float>(&variance)->default_value(10.0), "variance")

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


	const unsigned int Dimension = 3;

	//typedef float                              PixelType;
	//typedef itk::Image< PixelType, Dimension > ImageType;

	typedef itk::ImageFileReader<ImageType>  ReaderType;
	typedef itk::ImageFileWriter<ImageType> WriterType;

	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(input_file);
	reader->Update();

	ImageType::Pointer image = reader->GetOutput();

	ImageType::Pointer output_image = ImageType::New();
	output_image->SetRegions(image->GetLargestPossibleRegion());
	output_image->Allocate();

	//denoise(image, output_image);
	denoise(
		image->GetPixelContainer()->GetBufferPointer(),
		output_image->GetPixelContainer()->GetBufferPointer(),
		image->GetLargestPossibleRegion().GetSize()[0],
		image->GetLargestPossibleRegion().GetSize()[1],
		image->GetLargestPossibleRegion().GetSize()[2],
		variance);


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
	}

	return EXIT_SUCCESS;
}
