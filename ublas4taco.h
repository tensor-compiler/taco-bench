
#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef UBLAS
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>
typedef boost::numeric::ublas::compressed_matrix<double,boost::numeric::ublas::column_major> UBlasSparse;
typedef boost::numeric::ublas::vector<double> UBlasDenseVector;
#endif

  void tacoToUBLAS(const Tensor<double>& src, UBlasSparse& dst)
  {
    for (auto& value : iterate<double>(src))
      dst(value.first.at(0),value.first.at(1)) = value.second;
  }

  void UBLASTotaco(const UBlasDenseVector& src, Tensor<double>& dst){
    for (int i=0; i<dst.getDimension(0); ++i) {
      dst.insert({i}, src[i]);
    }
    dst.pack();
  }

  void tacoToUBLAS(const Tensor<double>& src, UBlasDenseVector& dst)
  {
    for (auto& value : iterate<double>(src))
      dst(value.first[0]) = value.second;
  }

  void exprToUBLAS(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        UBlasSparse Aublas(rows,cols);
        tacoToUBLAS(exprOperands.at("A"),Aublas);

        UBlasDenseVector xublas(cols), yublas(rows);
        tacoToUBLAS(exprOperands.at("x"),xublas);

        TACO_BENCH(boost::numeric::ublas::axpy_prod(Aublas, xublas, yublas, true);,"UBLAS",repeat,timevalue,true);

        Tensor<double> y_ublas({rows}, Dense);
        UBLASTotaco(yublas,y_ublas);

        validate("UBLAS", y_ublas, exprOperands.at("yRef"));
        break;
      }
      default:
        cout << " !! Expression not implemented for UBLAS" << endl;
        break;
    }
}
