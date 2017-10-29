#pragma once

#include <fstream>
#include <unordered_set>
#include <map>
#include <unordered_map>

#include <boost/spirit/home/x3.hpp>
#include <boost/format.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>

namespace x3 = boost::spirit::x3;
namespace ascii = boost::spirit::x3::ascii;

using namespace cugip;

struct Hash {
  size_t operator()(const Int2 &val) const {
	  return val[0] + val[1];
  }
};

struct EdgeStats {
	void add(float aVal1, float aVal2) {
		++count;
		sum += sqr(1.0f / (1.0f + std::max(aVal1, aVal2)));
	}

	float sum = 0.0f;
	int count = 0;
};

struct NodeStats {
	void add(float aValue) {
		++count;
		sum += aValue;
	}
	float sum = 0;
	int count = 0;
};

struct GraphStats {
	std::unordered_map<Int2, EdgeStats, Hash> edges;
	//std::unordered_map<int, NodeStats> nodes;
	std::map<int, NodeStats> nodes;
};

inline void saveGraph(const GraphStats &aGraph, const std::string &aOutputPath)
{
	std::ofstream file(aOutputPath, std::ofstream::out);
	for (const auto &node : aGraph.nodes) {
		file << "n;" << node.first << ';' << node.second.count << ';' << std::fixed << node.second.sum << '\n';
	}
	for (const auto &edge : aGraph.edges) {
		file << "e;" << edge.first[0] << ';' << edge.first[1] << ';' << edge.second.count << ';' << edge.second.sum << '\n';
	}
}

inline void parseAndAddNode(GraphStats &aGraph, const std::string &aLine)
{
	using x3::int_;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::tuple<int, int, float> result;
	bool const res = x3::phrase_parse(aLine.begin(), aLine.end(), "n;" >> int_ >> ';' >> int_ >> ';' >> float_, blank, result);
	aGraph.nodes[get<0>(result)] = NodeStats{ get<2>(result), get<1>(result) };
}

inline void parseAndAddEdge(GraphStats &aGraph, const std::string &aLine)
{
	using x3::int_;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::tuple<int, int, int, float> result;
	bool const res = x3::phrase_parse(aLine.begin(), aLine.end(), "e;" >> int_ >> ';' >> int_ >> ';' >> int_ >> ';' >> float_, blank, result);
	aGraph.edges[Int2(get<0>(result), get<1>(result))] = EdgeStats{ get<3>(result), get<2>(result) };
}

inline GraphStats loadGraph(const std::string &aInputPath)
{
	std::ifstream file(aInputPath, std::ifstream::in);
	GraphStats graph;
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		switch (line[0]) {
		case 'n':
			parseAndAddNode(graph, line);
			break;
		case 'e':
			parseAndAddEdge(graph, line);
			break;
		default:
			std::cout << "Unknown line identifier '" << line[0] << "' on line '" << line << "'\n";
		}
	}
	return graph;
}
