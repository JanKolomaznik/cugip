#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;


int main( int argc, char* argv[] )
{
	std::string input_file;
	std::string output_file;
	int iteration_count;
	float sigma;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&input_file), "input file")
		("output,o", po::value<std::string>(&output_file), "output file")
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

	typedef float                              PixelType;
	typedef itk::Image< PixelType, Dimension > ImageType;

	typedef itk::ImageFileReader<ImageType>  ReaderType;
	typedef itk::ImageFileWriter<ImageType> WriterType;

	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(input_file);
	reader->Update();

	ImageType::Pointer image = reader->GetOutput();


	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(output_file);
	writer->SetInput(image);

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



