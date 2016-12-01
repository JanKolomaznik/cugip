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
computeBoykovKolmogorov(
	const GraphStats &aGraph,
	const std::vector<std::pair<bool, int>> &aMarkers);

std::vector<int>
computeCudaGraphCut(
	const GraphStats &aGraph,
	const std::vector<std::pair<bool, int>> &aMarkers,
	CudacutSimpleConfig aConfig);


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
	using x3::lit;
	using ascii::blank;
	std::ifstream file(aPath.string(), std::ifstream::in);
	std::vector<std::pair<bool, int>> markers;
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		int result;

		bool const res = x3::phrase_parse(line.begin() + 1, line.end(), int_, blank, result);
		switch (line[0]) {
		case 'f':
			markers.emplace_back(true, result);
			break;
		case 'b':
			markers.emplace_back(false, result);
			break;
		default:
			std::cout << "Unknown marker line identifier '" << line[0] << "' on line '" << line << "'\n";
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
	float threshold;
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
		("residual-threshold,r", po::value<float>(&threshold)->default_value(0.0f), "residual threshold")
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
			outputs = computeBoykovKolmogorov(
				graph,
				markers
				);
			break;
		case Algorithm::CudaCut: {
				BOOST_LOG_TRIVIAL(info) << "Running CUDACut ...";
				CudacutSimpleConfig config(
					threshold
				);
				outputs = computeCudaGraphCut(
					graph,
					markers,
					config
					);
			}
			break;
		default:
			BOOST_LOG_TRIVIAL(error) << "Unknown algorithm";
		}

		BOOST_LOG_TRIVIAL(info) << "Saving output `" << outputFile << "` ...";
		std::ofstream file(outputFile.string(), std::ofstream::out);
		for (auto id : outputs) {
			file << id << '\n';
		}
	} catch (itk::ExceptionObject & error) {
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	} catch (...) {
		std::cerr << "Unknown error\n";
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
