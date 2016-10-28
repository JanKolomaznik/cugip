#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkSpatialObjectToImageFilter.h>
#include <itkEllipseSpatialObject.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <boost/format.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/image_locator.hpp>

#include <unordered_set>
#include <map>
#include <unordered_map>

using namespace cugip;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

typedef itk::Image<int, 3> LabelsType;
typedef itk::Image<uint8_t, 3> MaskType;

using SelectedRegions = std::unordered_set<int>;

template<typename TRegionsView, typename TMaskView>
SelectedRegions findMaskedRegions(TRegionsView aRegions, TMaskView aMask, uint8_t aMaskValue)
{
	simple_vector<int, 3> index;
	SelectedRegions ids;
	auto size = aMask.dimensions();
	for(index[2] = 0; index[2] < size[2]; ++index[2]) {
		for(index[1] = 0; index[1] < size[1]; ++index[1]) {
			for(index[0] = 0; index[0] < size[0]; ++index[0]) {
				//std::array<Int3, 3> offsets = {index, index, index};
				if (aMask[index] == aMaskValue) {
					auto label = aRegions[index];
					ids.insert(label);
				}
			}
		}
	}
	return ids;
}

int main( int argc, char* argv[] )
{
	std::string mask_file;
	std::string labels_file;
	std::string output_file;
	uint8_t maskValue;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("mask,m", po::value<std::string>(&mask_file), "input file")
		("labels,l", po::value<std::string>(&labels_file), "labels file")
		("value,v", po::value<uint8_t>(&maskValue)->default_value(255), "mask value")
		("output,o", po::value<std::string>(&output_file), "output file")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
	    std::cout << desc << "\n";
	    return 1;
	}

	if (vm.count("mask") == 0) {
	    std::cout << "Missing mask filename\n" << desc << "\n";
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
		maskReader->SetFileName(mask_file);
		maskReader->Update();
		MaskType::Pointer mask = maskReader->GetOutput();

		cugip::simple_vector<int, 3> size;
		for (int i = 0; i < 3; ++i) {
			size[i] = mask->GetLargestPossibleRegion().GetSize()[i];
		}

		auto labelsView = makeHostImageView(labels->GetPixelContainer()->GetBufferPointer(), size);
		auto maskView = makeHostImageView(mask->GetPixelContainer()->GetBufferPointer(), size);
		auto regions = findMaskedRegions(labelsView, maskView, maskValue);

		std::ofstream file(output_file, std::ofstream::out);
		for (auto id : regions) {
			file << id << '\n';
		}
	} catch( itk::ExceptionObject & error )	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
}
