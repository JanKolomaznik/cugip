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

typedef itk::Image<int, 3> ImageType;

struct Hash {
  size_t operator()(const Int2 &val) const {
	  return val[0] + val[1];
  }
};
template<typename TView>
std::unordered_set<Int2, Hash> outputGraphEdges(TView aView)
{
	std::unordered_set<Int2, Hash> edges;
	simple_vector<int, 3> index;
	auto size = aView.dimensions();
	for(index[2] = 0; index[2] < size[2] - 1; ++index[2]) {
		for(index[1] = 0; index[1] < size[1] - 1; ++index[1]) {
			for(index[0] = 0; index[0] < size[0] - 1; ++index[0]) {
				std::array<Int3, 3> offsets = {index, index, index};
				auto v1 = aView[index];
				for (int i = 0; i < 3; ++i) {
					auto index2 = index;
					index2[i] += 1;
					auto v2 = aView[index2];
					if (v1 != v2) {
						edges.insert(Int2(std::min(v1, v2), std::max(v1, v2)));
						//std::cout << v1 << ';' << v2 << ';' << edges.size() << '\n';
					}
				}
			}
		}
	}
	return edges;
}

template<typename TSet>
std::map<int, int> countEdges(const TSet &aEdges, std::string output_file)
{
	std::map<int, int> vertexDegrees;
	for (const auto &edge : aEdges) {
		vertexDegrees[edge[0]] += 1;
		vertexDegrees[edge[1]] += 1;
	}
	std::ofstream outFile(output_file);
	for (auto &rec : vertexDegrees) {
		outFile << rec.second << '\n';
	}
	return vertexDegrees;
}

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

	try
	{
		typedef itk::ImageFileReader<ImageType>  ReaderType;

		ReaderType::Pointer reader = ReaderType::New();
		reader->SetFileName(input_file);
		reader->Update();

		ImageType::Pointer image = reader->GetOutput();

		cugip::simple_vector<int, 3> size;
		for (int i = 0; i < 3; ++i) {
			size[i] = image->GetLargestPossibleRegion().GetSize()[i];
		}

		auto inView = makeHostImageView(image->GetPixelContainer()->GetBufferPointer(), size);
		auto edges = outputGraphEdges(inView);
		countEdges(edges, output_file);

	}
	catch( itk::ExceptionObject & error )
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
}
