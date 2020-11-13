#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

float myfunc(std::string path) {
  std::string line;
  std::vector<std::string> lines;
  std::ifstream myfile(path);
  if (myfile.is_open()) {
    while (std::getline(myfile, line)) {
      if ((line.find("user") != std::string::npos) ||
          (line.find("sys") != std::string::npos)) {
        lines.push_back(line);
      }
    }
    myfile.close();
  }
  assert(lines.size() == 2);
  std::vector<float> times;
  std::string token = "m";
  for (auto line : lines) {
    auto pos = line.find(token);
    auto sub = line.substr(pos + 1, line.length());
    sub.erase(std::remove(sub.begin(), sub.end(), 's'), sub.end());
    times.push_back(std::stof(sub));
  }
  assert(times.size() == 2);
  return times[0] + times[1];
}

int main(int argc, char *argv[]) {
  auto mlir = myfunc("./time.tactics.txt");
  auto clang = myfunc("time.mlir.txt"); 
  std::cout << "TACTICS " << mlir << " seconds\n";
  std::cout << "MLIR " << clang << " seconds\n";
  std::cout << " + " << ((mlir * 100.0)/clang) - 100 << "\n"; 
}
