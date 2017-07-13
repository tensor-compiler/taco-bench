#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef YOURS
// Add YOURS include here
// ...

  void exprToYOURS(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV:
        // Add code for YOURS implementation of SpMV
        // ..
        // Use TACO_BENCH macro to benchmark YOURS implementation
        // ..
        // Use validate method to compare against expected results
        // ..
      case PLUS3:
      case MATTRANSMUL:
      case RESIDUAL:
      case SDDMM:
      default:
        cout << " !! Expression not implemented for YOURS" << endl;
        break;
    }
  }

#endif
