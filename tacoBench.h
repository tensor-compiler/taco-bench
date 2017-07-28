
#include "taco.h"

using namespace taco;
using namespace std;

// MACRO to benchmark some CODE with REPEAT times and COLD/WARM cache
#define TACO_BENCH(CODE, NAME, REPEAT, TIMER, COLD) {           \
    TACO_TIME_REPEAT(CODE, REPEAT, TIMER, COLD);                \
    cout << NAME << " time (ms)" << endl << TIMER << endl;      \
}

#define CHECK_PRODUCT(NAME) {                                   \
    if (products.at(NAME)) {                                    \
      cout << "tacoBench was not compiled with "<< NAME << " and will not use it" << endl; \
      products.at(NAME)=false;                                  \
    }                                                           \
}


// Enum of possible expressions to Benchmark
enum BenchExpr {SpMV, PLUS3, MATTRANSMUL, RESIDUAL, SDDMM, SparsitySpMV, SparsityTTV};

// Compare two tensors of different formats
bool compare(const Tensor<double>&Dst, const Tensor<double>&Ref) {
  if (Dst.getDimensions() != Ref.getDimensions()) {
    return false;
  }

  std::set<std::vector<int>> coords;
  for (const auto& val : Dst) {
    if (!coords.insert(val.first).second) {
      return false;
    }
  }

  vector<std::pair<std::vector<int>,double>> valsDst;
  for (const auto& val : Dst) {
    if (val.second != 0) {
      valsDst.push_back(val);
    }
  }

  vector<std::pair<std::vector<int>,double>> valsRef;
  for (const auto& val : Ref) {
    if (val.second != 0) {
      valsRef.push_back(val);
    }
  }
  std::sort(valsRef.begin(), valsRef.end());
  std::sort(valsDst.begin(), valsDst.end());
  return valsDst == valsRef;
}

void validate (string name, const Tensor<double>& Dst, const Tensor<double>& Ref) {
  if (Dst.getFormat()==Ref.getFormat()) {
    if (!equals (Dst, Ref))
      cout << "\033[1;31m  Validation Error with " << name << " \033[0m" << endl;
  }
  else {
    if (!compare(Dst,Ref))
      cout << "\033[1;31m  Validation Error with " << name << " \033[0m" << endl;
  }
}
