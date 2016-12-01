#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkSpatialObjectToImageFilter.h>
#include <itkEllipseSpatialObject.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <boost/format.hpp>
#include <cugip/host_image_view.hpp>
#include <cugip/image_locator.hpp>

#include "../benchmarks/graph_cut/graph.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

typedef itk::Image<int, 3> LabelsType;
typedef itk::Image<float, 3> ImageType;


template<typename TRegionsView, typename TView>
GraphStats createGraph(TRegionsView aRegions, TView aView)
{
	//std::unordered_set<Int2, Hash> edges;
	GraphStats graph;
	simple_vector<int, 3> index;
	auto size = aView.dimensions();
	auto secondCorner = size - simple_vector<int, 3>(1, 1, 1);
	int maxLabel = 0;
	for(index[2] = 0; index[2] < size[2]; ++index[2]) {
		for(index[1] = 0; index[1] < size[1]; ++index[1]) {
			for(index[0] = 0; index[0] < size[0]; ++index[0]) {
				//std::array<Int3, 3> offsets = {index, index, index};
				auto label1 = aRegions[index];
				maxLabel = std::max(maxLabel, label1);
				auto value1 = aView[index];
				graph.nodes[label1].add(value1);
				if (index < secondCorner) {
					for (int i = 0; i < 3; ++i) {
						auto index2 = index;
						index2[i] += 1;
						auto label2 = aRegions[index2];
						if (label1 != label2) {
							auto value2 = aView[index2];
							auto edgeId = Int2(std::min(label1, label2), std::max(label1, label2));
							graph.edges[edgeId].add(value1, value2);
							//std::cout << v1 << ';' << v2 << ';' << edges.size() << '\n';
						}
					}
				}
			}
		}
	}
	std::cout << "Max label " << maxLabel << "\n";
	return graph;
}

/*template<typename TSet>
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
}*/

int main( int argc, char* argv[] )
{
	std::string input_file;
	std::string labels_file;
	std::string output_file;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("input,i", po::value<std::string>(&input_file), "input file")
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
		typedef itk::ImageFileReader<ImageType>  ImageReaderType;

		LabelsReaderType::Pointer labelsReader = LabelsReaderType::New();
		labelsReader->SetFileName(labels_file);
		labelsReader->Update();
		LabelsType::Pointer labels = labelsReader->GetOutput();

		ImageReaderType::Pointer imageReader = ImageReaderType::New();
		imageReader->SetFileName(input_file);
		imageReader->Update();
		ImageType::Pointer image = imageReader->GetOutput();

		cugip::simple_vector<int, 3> size;
		for (int i = 0; i < 3; ++i) {
			size[i] = image->GetLargestPossibleRegion().GetSize()[i];
		}

		auto labelsView = makeHostImageView(labels->GetPixelContainer()->GetBufferPointer(), size);
		auto imageView = makeHostImageView(image->GetPixelContainer()->GetBufferPointer(), size);
		auto graph = createGraph(labelsView, imageView);
		std::cout << "Node count " << graph.nodes.size() << "\n";
		std::cout << "Edge count " << graph.edges.size() << "\n";
		saveGraph(graph, output_file);
		//auto edges = outputGraphEdges(inView);
		//countEdges(edges, output_file);

	}
	catch( itk::ExceptionObject & error )
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
}
