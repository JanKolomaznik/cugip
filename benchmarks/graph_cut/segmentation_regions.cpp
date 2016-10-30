#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/filesystem.hpp>
#include <cugip/itk_utils.hpp>

#include <boost/log/trivial.hpp>
#include <string>

#include "graph_cut_trace_utils.hpp"

#include "graph.hpp"

namespace po = boost::program_options;

typedef itk::Image<float, 3> ImageType;
typedef itk::Image<int, 3> IntImageType;
typedef itk::Image<uint8_t, 3> MarkersType;
typedef itk::Image<uint8_t, 3> MaskType;

std::vector<int>
computeBoykovKolmogorovGrid(
	const GraphStats &aGraph,
	const std::vector<int> &aMarkers);


/*void
computeCudaGraphCut(
	cugip::const_host_image_view<const float, 3> aData,
	cugip::const_host_image_view<const uint8_t, 3> aMarkers,
	cugip::host_image_view<uint8_t, 3> aOutput,
	float aSigma,
	uint8_t aMaskValue,
	CudacutConfig &aConfig);*/

enum class Algorithm {
	BoykovKolmogorov,
	CudaCut
};

Algorithm getAlgorithmFromString(const std::string &token)
{
	if (token == "boykov-kolmogorov") {
		return Algorithm::BoykovKolmogorov;
	}  else if (token == "cudacut") {
		return Algorithm::CudaCut;
	} else {
		throw po::validation_error(po::validation_error::invalid_option_value);
	}
	return Algorithm::BoykovKolmogorov;
}

std::vector<std::pair<bool, int>> loadMarkers(const boost::filesystem::path &aPath)
{
	using x3::int_;
	using x3::char_;
	using ascii::blank;
	std::ifstream file(aPath, std::ifstream::in);
	std::vector<std::pair<bool, int>> markers;
	std::string line;
	while (file >> line) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		std::tuple<char, int> result;

		bool const res = x3::phrase_parse(aLine.begin(), aLine.end(), char_ >> int_, blank, result);
		switch (std::get<0>(result)) {
		case 'f':
			markers.emplace_back(true, std:get<1>(result));
			break;
		case 'b':
			markers.emplace_back(false, std:get<1>(result));
			break;
		default:
			std::cout << "Unknown line identifier '" << std::get<0>(result) << "'\n";
		}
	}
	return markers;
}

int
main(int argc, char* argv[])
{
	boost::filesystem::path inputFile;
	boost::filesystem::path markersFile;
	boost::filesystem::path outputFile;
	float sigma;
	uint8_t maskValue;

	Algorithm algorithm;
	std::string algorithmName;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<boost::filesystem::path>(&inputFile), "graph file")
		("markers,m", po::value<boost::filesystem::path>(&markersFile), "markers file")
		("algorithm,a", po::value<std::string>(&algorithmName)->default_value("boykov-kolmogorov"), "boykov-kolmogorov, cudacut")
		("output,o", po::value<boost::filesystem::path>(&outputFile), "output file")
		//("sigma,s", po::value<float>(&sigma)->default_value(1.0f), "input noise deviation")
		//("mask-value,v", po::value<uint8_t>(&maskValue)->default_value(255), "mask value")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	algorithm = getAlgorithmFromString(algorithmName);

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

	if (vm.count("output") == 0) {
		std::cout << "Missing output filename\n" << desc << "\n";
		return 1;
	}

	try {
		BOOST_LOG_TRIVIAL(info) << "Loading inputs ...";
		auto graph = loadGraph(inputFile.string());
		auto markers = loadMarkers(markersFile.string());
		std::vector<int> outputs;
		switch (algorithm) {
		case Algorithm::BoykovKolmogorov:
			BOOST_LOG_TRIVIAL(info) << "Running Boykov-Kolmogorov graph cut ...";
			outputs = computeBoykovKolmogorovGrid(
				graph,
				markers
				);
			break;
		case Algorithm::CudaCut: {
				BOOST_LOG_TRIVIAL(info) << "Running CUDACut ...";
				/*boost::filesystem::path outputDir = outputFile.parent_path();
				std::string stem = outputFile.stem().string();
				MaskType::Pointer saturated = MaskType::New();
				saturated->SetRegions(image->GetLargestPossibleRegion());
				saturated->Allocate();
				saturated->SetSpacing(image->GetSpacing());

				MaskType::Pointer excess = MaskType::New();
				excess->SetRegions(image->GetLargestPossibleRegion());
				excess->Allocate();
				excess->SetSpacing(image->GetSpacing());

				ImageType::Pointer label = ImageType::New();
				label->SetRegions(image->GetLargestPossibleRegion());
				label->Allocate();
				label->SetSpacing(image->GetSpacing());

				CudacutConfig config(
					cugip::view(*(saturated.GetPointer())),
					cugip::view(*(excess.GetPointer())),
					cugip::view(*(label.GetPointer()))
				);

				computeCudaGraphCut(
					cugip::const_view(*(image.GetPointer())),
					cugip::const_view(*(markers.GetPointer())),
					cugip::view(*(outputImage.GetPointer())),
					sigma,
					maskValue,
					config);
				saveImage<MaskType>(saturated, outputDir / (stem + "_saturated.mrc"));
				saveImage<MaskType>(excess, outputDir / (stem + "_excess.mrc"));
				saveImage<ImageType>(label, outputDir / (stem + "_label.mrc"));*/
			}
			break;
		default:
			BOOST_LOG_TRIVIAL(error) << "Unknown algorithm";
		}

		BOOST_LOG_TRIVIAL(info) << "Saving output `" << outputFile << "` ...";
	} catch (itk::ExceptionObject & error) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
