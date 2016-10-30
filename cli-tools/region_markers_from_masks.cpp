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

typedef itk::Image<int, 3> LabelsType;
typedef itk::Image<float, 3> ImageType;
typedef itk::Image<uint8_t, 3> MaskType;


template<typename TRegionsView, typename TMaskView>
generateMarkers(TRegionsView aRegions, TMaskView aMask, std::set<int> &aForeground, std::set<int> &aBackground)
{
	simple_vector<int, 3> index;
	auto size = aMask.dimensions();
	for(index[2] = 0; index[2] < size[2]; ++index[2]) {
		for(index[1] = 0; index[1] < size[1]; ++index[1]) {
			for(index[0] = 0; index[0] < size[0]; ++index[0]) {
				if (aMask[index] == 255) {
					aForeground.insert(aRegions[index]);
				} else if (aMask[index] == 128) {
					aBackground.insert(aRegions[index]);
				}
			}
		}
	}
}

int main( int argc, char* argv[] )
{
	std::string input_file;
	std::string labels_file;
	std::string output_file;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&input_file), "input mask file")
		("labels,l", po::value<std::string>(&labels_file), "labels file")
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

	if (vm.count("labels") == 0) {
	    std::cout << "Missing labels filename\n" << desc << "\n";
	    return 1;
	}

	if (vm.count("output") == 0) {
	    std::cout << "Missing output filename\n" << desc << "\n";
	    return 1;
	}


	const unsigned int Dimension = 3;

	try
	{
		typedef itk::ImageFileReader<LabelsType>  LabelsReaderType;
		typedef itk::ImageFileReader<MaskType>  MaskReaderType;

		LabelsReaderType::Pointer labelsReader = LabelsReaderType::New();
		labelsReader->SetFileName(labels_file);
		labelsReader->Update();
		LabelsType::Pointer labels = labelsReader->GetOutput();

		MaskReaderType::Pointer maskReader = MaskReaderType::New();
		maskReader->SetFileName(input_file);
		maskReader->Update();
		MaskType::Pointer mask = maskReader->GetOutput();

		cugip::simple_vector<int, 3> size;
		for (int i = 0; i < 3; ++i) {
			size[i] = image->GetLargestPossibleRegion().GetSize()[i];
		}

		auto labelsView = makeHostImageView(labels->GetPixelContainer()->GetBufferPointer(), size);
		auto maskView = makeHostImageView(mask->GetPixelContainer()->GetBufferPointer(), size);
		std::set<int> foreground;
		std::set<int> background;
		generateMarkers(labelsView, maskView, foreground, background);

		std::ofstream file(output_file, std::ofstream::out);
		for (auto label : foreground) {
			file << "f " << label << '\n';
		}
		for (auto label : background) {
			file << "b " << label << '\n';
		}
	}
	catch( itk::ExceptionObject & error )
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
}
