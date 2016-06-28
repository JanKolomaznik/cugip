#include <fstream>
#include <boost/format.hpp>

namespace cugip {

void
Graph::save_to_graphml(const boost::filesystem::path &file) const
{
	std::fstream out(file.string().c_str(), std::ios_base::out);


	thrust::host_vector<EdgeWeight> excess = mExcess;
	thrust::host_vector<int> labels = mLabels;

	thrust::host_vector<EdgeRecord> edges = mEdgeDefinitions;
	thrust::host_vector<EdgeWeight> weights = mEdgeWeightsForward;
	thrust::host_vector<EdgeWeight> weightsBackward = mEdgeWeightsBackward; //TODO - backward capacity

	thrust::host_vector<EdgeWeight> source_links = mSourceTLinks;
	thrust::host_vector<EdgeWeight> sink_links = mSinkTLinks;

	out <<	"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
		"<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n";
		/*"xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
		"xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns"
		"http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n";*/

	out << "<key id=\"capacity\" for=\"edge\" attr.name=\"capacity\" attr.type=\"float\"/>\n";

	out << "<node id=\"source\"/>\n";
	out << "<node id=\"sink\"/>\n";

	for (int i = 0; i < excess.size(); ++i) {
		out << boost::format("<node id=\"n%1%\"/>\n") % i;
	}
	for (int i = 0; i < edges.size(); ++i) {
		out <<
			boost::format("<edge source=\"n%1%\" target=\"n%2%\">\n\t<data key=\"capacity\">%3%</data>\n</edge>\n")
				% edges[i].first
				% edges[i].second
				% weights[i];
	}

	for (int i = 0; i < source_links.size(); ++i) {
		if (source_links[i] > 0.0f) {
			out << boost::format("<edge source=\"source\" target=\"n%1%\">\n\t<data key=\"capacity\">%2%</data>\n</edge>\n")
				% i
				% source_links[i];
		}
	}
	for (int i = 0; i < sink_links.size(); ++i) {
		if (sink_links[i] > 0.0f) {
			out << boost::format("<edge source=\"sink\" target=\"n%1%\">\n\t<data key=\"capacity\">%2%</data>\n</edge>\n")
				% i
				% sink_links[i];
		}
	}


	out << "</graphml>\n";
}

}//namespace cugip

