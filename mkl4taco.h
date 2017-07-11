
#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef MKL
  #include "mkl_spblas.h"

  void exprToMKL(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV: {
        char matdescra[6] = "G  C ";
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        int ptrsize = exprOperands.at("A").getStorage().getIndex().getSize();
        double *a_CSC;
        int* ia_CSC;
        int* ja_CSC;
        getCSCArrays(exprOperands.at("A"),&ia_CSC,&ja_CSC,&a_CSC);
        int* pointerB=new int[ptrsize-1];
        int* pointerE=new int[ptrsize-1];
        for (int i=0; i<ptrsize-1; i++) {
          pointerB[i]=ia_CSC[i];
          pointerE[i]=ia_CSC[i+1];
        }
        Tensor<double> y_mkl({rows}, Dense);
        y_mkl.pack();

        double malpha=1.0;
        double mbeta=0.0;
        char transa = 'N';
        TACO_BENCH(mkl_dcscmv(&transa, &rows, &cols, &malpha, matdescra, a_CSC, ja_CSC, pointerB,
                              pointerE, (double*)(exprOperands.at("x").getStorage().getValues().getData()),
                              &mbeta, (double*)(y_mkl.getStorage().getValues().getData()));,
                   "MKL", repeat,timevalue,true)

        validate("MKL", y_mkl, exprOperands.at("yRef"));

        break;
      }
      default:
        cout << " !! Expression not implemented for MKL" << endl;
        break;
    }
  }

#endif
