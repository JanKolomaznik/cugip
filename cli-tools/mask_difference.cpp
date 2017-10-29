#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkSpatialObjectToImageFilter.h>
#include <itkEllipseSpatialObject.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <boost/format.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/image_locator.hpp>

#include <set>

#include "../benchmarks/graph_cut/graph.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

typedef itk::Image<uint8_t, 3> MaskType;


template<typename TMaskView>
std::pair<int, int> computeMaskDifference(TMaskView aMask1, TMaskView aMask2)
{
	simple_vector<int, 3> index;
	auto size = aMask1.dimensions();
	int difference = 0;
	int count = 0;
	for(index[2] = 0; index[2] < size[2]; ++index[2]) {
		for(index[1] = 0; index[1] < size[1]; ++index[1]) {
			for(index[0] = 0; index[0] < size[0]; ++index[0]) {
				if (aMask1[index] != aMask2[index]) {
					++difference;
				}
				if (aMask1[index] == 255) {
					++count;
				}
			}
		}
	}
	return { difference, count };
}

int main( int argc, char* argv[] )
{
	std::string input_file1;
	std::string input_file2;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&input_file1), "input mask file")
		("second,s", po::value<std::string>(&input_file2), "second mask file")
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

	if (vm.count("second") == 0) {
	    std::cout << "Missing second mask filename\n" << desc << "\n";
	    return 1;
	}

	const unsigned int Dimension = 3;

	try
	{
		typedef itk::ImageFileReader<MaskType>  MaskReaderType;

		MaskReaderType::Pointer maskReader1 = MaskReaderType::New();
		maskReader1->SetFileName(input_file1);
		maskReader1->Update();
		MaskType::Pointer mask1 = maskReader1->GetOutput();

		MaskReaderType::Pointer maskReader2 = MaskReaderType::New();
		maskReader2->SetFileName(input_file2);
		maskReader2->Update();
		MaskType::Pointer mask2 = maskReader2->GetOutput();

		cugip::simple_vector<int, 3> size;
		for (int i = 0; i < 3; ++i) {
			size[i] = mask1->GetLargestPossibleRegion().GetSize()[i];
		}

		auto maskView1 = makeHostImageView(mask1->GetPixelContainer()->GetBufferPointer(), size);
		auto maskView2 = makeHostImageView(mask2->GetPixelContainer()->GetBufferPointer(), size);
		auto difference = computeMaskDifference(maskView1, maskView2);

		std::cout << "Mask difference " << difference.first << " / " << difference.second << '\n';
	}
	catch( itk::ExceptionObject & error )
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
}
