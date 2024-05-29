#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>
#include <fmt/core.h>

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

float computeFlops(std::string name, float time) {
  if (name.compare("atax") == 0)
    return 15960000.0 / time / 1.0E9;
  if (name.compare("2mm") == 0) 
    return 3312000000.0 / time / 1.0E9;
  if (name.compare("3mm") == 0) 
    return 5400000000.0 / time / 1.0E9;
  if (name.compare("ab-acd-dbc") == 0)
    return 2147483648.0 / time / 1.0E9;
  if (name.compare("ab-cad-dcb") == 0)
    return 2147483648.0 / time / 1.0E9;
  if (name.compare("abc-acd-db") == 0)
    return 2147483648.0 / time / 1.0E9;
  if (name.compare("abc-ad-bdc") == 0)
    return 2147483648.0 / time / 1.0E9;
  if (name.compare("abc-bda-dc") == 0)
    return 2147483648.0 / time / 1.0E9;
  if (name.compare("abcd-aebf-dfce") == 0)
    return 2147483648.0 / time / 1.0E9;
  if (name.compare("abcd-aebf-fdec") == 0)
    return 2147483648.0 / time/ 1.0E9;
  if (name.compare("bicg") == 0)
    return 15960000.0 / time/ 1.0E9;
  if (name.compare("gemm") == 0)
    return 2640000000.0 / time / 1.0E9;
  if (name.compare("gemver") == 0)
    return 40000000.0 / time / 1.0E9;
  if (name.compare("gesummv") == 0)
    return 6763900.0 / time / 1.0E9;
  if (name.compare("mvt") == 0)
    return 16000000.0 / time / 1.0E9;
  if (name.compare("conv2d-nchw") == 0)
    return 8589934592.0 / time / 1.0E9;
  std::cout << "flops number not available for: " << name << std::endl;
  assert(0);
}

void buildBarPlot(const std::vector<std::string> &all) {
  assert(all.size() == 4);
  
  std::string s = fmt::format(R"(
  \documentclass{{article}}
    \usepackage[utf8]{{inputenc}}
    \usepackage{{pgfplots}}
    \usepackage{{color}}
    
    \definecolor{{orange3}}{{rgb}}{{0.808,0.361,0.000}}
    \definecolor{{aluminium2}}{{rgb}}{{0.827,0.843,0.812}}
    \definecolor{{blind_safe_one_scheme_four_colors}}{{RGB}}{{166,206,227}}
    \definecolor{{blind_safe_two_scheme_four_colors}}{{RGB}}{{31,120,180}}
    \definecolor{{blind_safe_three_scheme_four_colors}}{{RGB}}{{178,223,138}}

    \begin{{document}}
    \title{{Experiment Section 5.2}}
    
    \begin{{figure*}}
    \pgfplotsset{{compat = 1.3}}
    \pgfplotsset{{major grid style={{dotted,aluminium2!50!black}}}}
    \begin{{tikzpicture}}
    \begin{{axis}}[
      width=1\textwidth,
      every axis plot post/.style={{/pgf/number format/fixed}},
      ybar=1pt,
      enlargelimits=0.028,
      ylabel={{GFLOP/sec}},
      ymode=log,
      log basis y={{2}},
      bar width=2pt,
      legend columns=5,
      height=0.5\textwidth,
      ymajorgrids=true,
      grid style=dashed,
      log ticks with fixed point,
      axis x line*=bottom,
      x tick label style={{xshift=.4em,rotate=45,anchor=east}},
      ytick={{1, 4, 8, 16, 32, 64, 128, 256}},
      yminorticks=true,
      legend style={{at={{(1,1.2)}},anchor=north east}},
      xlabel={{}},
      extra y ticks={{1, 4, 16, 32, 64, 128, 256}},
      extra y tick labels={{}}, % do not print extra tick
      symbolic x coords={{atax, bicg, mvt, gemver, gesummv, 2mm, 3mm, conv2d-nchw, gemm, ab-acd-dbc, abc-acd-db, abc-ad-bdc, ab-cad-dcb, abc-bda-dc, abcd-aebf-dfce, abcd-aebf-fdec, geomean}},
      xtick=data,
  ]
    \addplot+ [orange3] coordinates {{
      {0}
    }};
  
    \addplot+ [blind_safe_three_scheme_four_colors] coordinates {{
      {1}
    }};
    
    \addplot+ [blind_safe_one_scheme_four_colors] coordinates {{
      {2}
    }};
    \addplot+ [blind_safe_two_scheme_four_colors] coordinates {{
      {3}
    }};
    \legend{{Clang -O3, Pluto, MLT-Linalg, MLT-Blas}}
    \end{{axis}}
    \end{{tikzpicture}}
    \end{{figure*}}
  \end{{document}}
  )", all[0], all[1], all[2], all[3]);

  std::cout << s << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 5)
    return 0;
  
  std::vector<std::string> allFormattedResults;
  for (int i = 1; i < argc; i++) {
    std::vector<std::string> lines;
    std::string path = argv[i]; 
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

    std::string formattedResults;
    std::vector<float> all;
    for (const auto &r : resFiltered) {
      float flops = computeFlops(r.first, r.second);
      std::string formattedResult = "(" + r.first + ", " + std::to_string(flops) + ") ";
      formattedResults += formattedResult;
      all.push_back(flops);
    }
    auto geomean = computeGeoMean(all);
    std::string formattedGeomean = "(geomean, " + std::to_string(geomean) + ")\n";
    formattedResults += formattedGeomean;
    allFormattedResults.push_back(formattedResults);
  }
  buildBarPlot(allFormattedResults);
}
