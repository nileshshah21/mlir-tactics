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

vecFiltered filter(const vec &vector) {
  vecFiltered res{};
  for (const auto &v : vector) {
    float min = findMin(std::get<1>(v));
    res.push_back(std::make_pair(std::get<0>(v), min));
  }
  return res;
}

void buildTable(const vecFiltered &vector) {
  assert(vector.size() == 6);

  std::string s = fmt::format(R"(
  \documentclass{{article}}
  \usepackage[utf8]{{inputenc}}
  \usepackage{{booktabs}}
  \usepackage{{graphicx}}

  \begin{{document}}
  \title{{Experiment Section 5.3}}

  \begin{{table*}}
  \begin{{center}}
  \resizebox{{\textwidth}}{{!}}{{%
  \begin{{tabular}}{{lllllll}}
    \toprule
    \footnotesize{{N}} & 
    \footnotesize{{Matrix Dimensions}} & 
    \footnotesize{{Initial Parenthesization (IP)}} & 
    \footnotesize{{Optimal Parenthesization (OP)}} &
    \footnotesize{{Time IP}} &
    \footnotesize{{Time OP}} & 
    \footnotesize{{Speedup}} \\ \midrule

    \footnotesize{{4}} & 
    \footnotesize{{800 1100 900 1200 100}} &
    \footnotesize{{$(((A_1 \times A_2) \times A_3) \times A_4)$}} &
    \footnotesize{{$(A_1 \times (A_2 \times (A_3 \times A_4)))$}} &
    \footnotesize{{{0} s}} & 
    \footnotesize{{{1} s}} & 
    \footnotesize{{{2} X}} \\

    \footnotesize{{5}} & 
    \footnotesize{{1000 2000 900 1500 600 800}} &
    \footnotesize{{$((((A_1 \times A_2) \times A_3) \times A_4) \times A_5)$}} &
    \footnotesize{{$((A_1 \times ( A_2 \times ( A_3 \times A_4))) \times A_5)$}} &
    \footnotesize{{{3} s}} & 
    \footnotesize{{{4} s}} & 
    \footnotesize{{{5}X}} \\

    \footnotesize{{6}} & 
    \footnotesize{{1500 400 2000 2200 600 1400 1000}} &
    \footnotesize{{$(((((A_1 \times A_2) \times A_3) \times A_4) \times A_5) \times A_6)$}} &
    \footnotesize{{$(A_1 \times ((((A_2 \times A_3) \times A_4) \times A_5) \times A_6))$}} &
    \footnotesize{{{6} s}} & 
    \footnotesize{{{7} s}} & 
    \footnotesize{{{8} X}} \\ \bottomrule 

  \end{{tabular}} 
  }}
  \end{{center}}
  \end{{table*}}
  \end{{document}}
)", vector[0].second, vector[1].second, vector[0].second/vector[1].second,
    vector[2].second, vector[3].second, vector[2].second/vector[3].second,
    vector[4].second, vector[5].second, vector[4].second/vector[5].second);

  std::cout << s << std::endl;
}


int main(int argc, char *argv[]) {
  if (argc != 4)
    return 0;

  std::vector<std::string> lines{};
  for (int i = 1; i < argc; i++) {
    std::string path = argv[i]; 
    std::string line;
    std::ifstream myfile(path);
    if (myfile.is_open()) {
      while (std::getline(myfile, line))
        lines.push_back(line);
      myfile.close();
    }
  } 
  vec res{};
  res = split(lines);
  vecFiltered resFiltered{};
  resFiltered = filter(res);
  buildTable(resFiltered);
    
}
