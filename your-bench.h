#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef YOUR
// Add your include here
// ...

  void exprToYour(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV:
        // Add code for your implementation of SpMV
        // ..
        // Use TACO_BENCH macro to benchmark your implementation
        // ..
        // Use validate method to compare against expected results
        // ..
      case PLUS3:
      case MATTRANSMUL:
      case RESIDUAL:
      case SDDMM:
      default:
        cout << " !! Expression not implemented for your" << endl;
        break;
    }
  }

#endif
