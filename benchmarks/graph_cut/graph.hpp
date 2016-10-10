#pragma once

#include <fstream>
#include <unordered_set>
#include <map>
#include <unordered_map>

using namespace cugip;

struct Hash {
  size_t operator()(const Int2 &val) const {
	  return val[0] + val[1];
  }
};

struct EdgeStats {
	void add(float aVal1, float aVal2) {}

	float sum = 0.0f;
	int count = 0;
};

struct NodeStats {
	void add(float aValue) {}
	float sum = 0;
	int count = 0;
};

struct GraphStats {
	std::unordered_map<Int2, EdgeStats, Hash> edges;
	std::unordered_map<int, NodeStats> nodes;
};

inline void saveGraph(const GraphStats &aGraph, const std::string &aOutputPath)
{
	std::ofstream file(aOutputPath, std::ofstream::out);
	for (const auto &node : aGraph.nodes) {
		file << "n;" << node.first << ';' << node.second.count << ';' << node.second.sum << '\n';
	}
	for (const auto &edge : aGraph.edges) {
		file << "e;" << edge.first[0] << ';' << edge.first[2] << ';' << edge.second.count << ';' << edge.second.sum << '\n';
	}
}

inline void parseAndAddNode(GraphStats &aGraph, const std::string &aLine)
{

}

inline void parseAndAddEdge(GraphStats &aGraph, const std::string &aLine)
{

}

inline GraphStats loadGraph(const std::string &aInputPath)
{
	std::ifstream file(aInputPath, std::ifstream::in);
	GraphStats graph;
	std::string line;
	while (file >> line) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		switch (line[0]) {
		case 'n':
			break;
		case 'e':
			break;
		default:
			std::cout << "Unknown line identifier '" << line[0] << "'\n";
		}
	}
	return graph;
}
