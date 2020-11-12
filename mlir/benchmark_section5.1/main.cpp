#include <iostream>
#include <fmt/core.h>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

using vec = std::vector<std::pair<std::string, int>>;
vec split(const std::vector<std::string> &lines) {
  vec res{};
  std::string delimiter = ":";

  for (const auto &line : lines) {
    if (auto pos = line.find(delimiter)) {
      size_t size = std::distance(line.begin(), line.end());
      auto tokenName = line.substr(0, pos);
      auto tokenValue = std::stoi(line.substr(pos+1, size));
      res.push_back(std::make_pair(tokenName, tokenValue));
    }
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
  auto vect = split(lines);

  std::string result;
  for (const auto &vec : vect) {
    result += "(" + vec.first + ", " + std::to_string(vec.second) + ") ";
  } 

  std::string s = fmt::format(R"( 
  \documentclass{{article}}
    \usepackage[utf8]{{inputenc}}
    \usepackage{{pgfplots}}
    \usepackage{{color}}
    
    \definecolor{{blind_safe_two_scheme_three_colors}}{{RGB}}{{252,141,98}}
    \definecolor{{blind_safe_three_scheme_three_colors}}{{RGB}}{{141,160,203}}
    \definecolor{{aluminium2}}{{rgb}}{{0.827,0.843,0.812}}

    \begin{{document}}
    \title{{Experiment Section 5.1}}

    \begin{{figure}}
    \pgfplotsset{{compat = 1.3}}
    \pgfplotsset{{major grid style={{dotted,aluminium2!50!black}}}}
    \begin{{tikzpicture}}
    \begin{{axis}}[
      width=0.5\textwidth,
      every axis plot post/.style={{/pgf/number format/fixed}},
      ybar=2pt,
      ylabel={{Callsites}},
      legend style={{draw=none, fill=none}},
      bar width=6pt,
      axis x line*=bottom,
      legend columns=3,
      ymin=0, ymax=4,
      height=0.5\textwidth,
      ymajorgrids=true,
      grid style=dashed,
      log ticks with fixed point,
      x tick label style={{xshift=4pt,rotate=45,left}},
      xlabel={{ }},
      symbolic x coords={{mm, 2mm, 3mm, darknet}},
      xtick=data,
    ]
        
    \addplot+ [blind_safe_three_scheme_three_colors] coordinates {{
      {0} 
    }};
    
    \addplot+ [blind_safe_two_scheme_three_colors] coordinates {{
      (mm, 1) (2mm, 2) (3mm, 3) (darknet, 1)
    }};
   
    \legend{{Detected, Oracle}}
    \end{{axis}}
    \end{{tikzpicture}}
    \caption{{Number of callsites detected by Multi-Level Tactics compared to perfect matching (Oracle).}}
    \end{{figure}}
  \end{{document}}
  )", result);

  std::cout << s << std::endl;
  return 0;
}
