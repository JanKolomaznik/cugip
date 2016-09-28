
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/program_options.hpp>

#include <cugip/host_image_view.hpp>
//#include <cugip/for_each.hpp>

namespace po = boost::program_options;

typedef itk::Image<int, 3> ImageType;

using namespace cugip;

void relabeling(
	host_image_view<int64_t, 3> aInput,
	int64_t aStart);

int main( int argc, char* argv[] )
{
	std::string input_file;
	std::string output_file;

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

	typedef itk::ImageFileReader<ImageType>  ReaderType;
	typedef itk::ImageFileWriter<ImageType> WriterType;

	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(input_file);
	reader->Update();

	ImageType::Pointer image = reader->GetOutput();

	cugip::simple_vector<int, 3> size;
	for (int i = 0; i < 3; ++i) {
		size[i] = image->GetLargestPossibleRegion().GetSize()[i];
	}

	auto inView = makeHostImageView(image->GetPixelContainer()->GetBufferPointer(), size);
	relabeling(inView, 1);

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
