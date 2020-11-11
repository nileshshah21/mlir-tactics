#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

using vec = std::vector<std::tuple<std::string, std::vector<float>>>;
using vecFiltered = std::vector<std::pair<std::string, float>>;

vec split(const std::vector<std::string> &lines) {
  vec res{};
  std::string delimiter = ":";

  for (const auto &line : lines) {
    std::string tmpLine = line;
    size_t pos = 0;
    std::string token;

    bool isString = true;
    std::string id;
    std::vector<float> times{};

    while ((pos = tmpLine.find(delimiter)) != std::string::npos) {
      token = tmpLine.substr(0, pos);
      if (isString) {
        isString = false;
        id = token;
      } else {
        times.push_back(std::stof(token));
      }
      tmpLine.erase(0, pos + delimiter.length());
    }

    res.push_back(std::make_tuple(id, times));
  }
  return res;
}

float findMin(const std::vector<float> &all) {
  auto it = std::min_element(all.begin(), all.end());
  return all.at(std::distance(all.begin(), it));
}

double computeGeoMean(const std::vector<float> &data) {

  auto product = 1.0;
  for (const auto x : data)
    product *= x;
  return std::pow(product, 1.0/data.size());
}

vecFiltered filter(const vec &vector) {
  vecFiltered res{};
  for (const auto &v : vector) {
    float min = findMin(std::get<1>(v));
    res.push_back(std::make_pair(std::get<0>(v), min));
  }
  return res;
}

int main(int argc, char *argv[]) {
  if (argc != 2)
    return 0;
  std::string path = argv[1];

  std::vector<std::string> lines{};
  std::string line;
  std::ifstream myfile(path);
  if (myfile.is_open()) {
    while (std::getline(myfile, line))
      lines.push_back(line);
    myfile.close();
  }
  vec res{};
  res = split(lines);
  vecFiltered resFiltered{};
  resFiltered = filter(res);

  std::vector<float> all;
  for (const auto &r : resFiltered) {
    std::cout << "(" << r.first << ", " << r.second << ")\n";
    all.push_back(r.second);
  }
  auto geomean = computeGeoMean(all);
  std::cout << "\n\nGeomean: " << geomean << "\n";
}
