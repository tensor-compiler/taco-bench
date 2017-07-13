#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef UBLAS
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix.hpp>
typedef boost::numeric::ublas::compressed_matrix<double,boost::numeric::ublas::column_major> UBlasSparse;
typedef boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major> UBlasColMajor;
typedef boost::numeric::ublas::matrix<double,boost::numeric::ublas::row_major> UBlasRowMajor;
typedef boost::numeric::ublas::vector<double> UBlasDenseVector;
#endif

  void UBLASTotaco(const UBlasSparse& src, Tensor<double>& dst){
    for (auto it1 = src.begin2(); it1 != src.end2(); it1++ )
      for (auto it2 = it1.begin(); it2 != it1.end(); ++it2 )
        dst.insert({(int)(it2.index1()),(int)(it2.index2())},*it2);
    dst.pack();
  }

  void tacoToUBLAS(const Tensor<double>& src, UBlasSparse& dst) {
    for (auto& value : iterate<double>(src))
      dst(value.first.at(0),value.first.at(1)) = value.second;
  }

  void tacoToUBLAS(const Tensor<double>& src, UBlasColMajor& dst) {
    dst.resize(src.getDimension(0), src.getDimension(1), false);
    for (auto& value : iterate<double>(src))
      dst(value.first.at(0),value.first.at(1)) = value.second;
  }

  void tacoToUBLAS(const Tensor<double>& src, UBlasRowMajor& dst) {
    dst.resize(src.getDimension(0), src.getDimension(1), false);
    for (auto& value : iterate<double>(src))
      dst(value.first.at(0),value.first.at(1)) = value.second;
  }

  void UBLASTotaco(const UBlasDenseVector& src, Tensor<double>& dst){
    for (int i=0; i<dst.getDimension(0); ++i)
      dst.insert({i}, src[i]);
    dst.pack();
  }

  void tacoToUBLAS(const Tensor<double>& src, UBlasDenseVector& dst)  {
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
      case PLUS3: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        UBlasSparse Aublas(rows,cols);
        UBlasSparse Bublas(rows,cols);
        UBlasSparse Cublas(rows,cols);
        UBlasSparse Dublas(rows,cols);

        tacoToUBLAS(exprOperands.at("B"),Bublas);
        tacoToUBLAS(exprOperands.at("C"),Cublas);
        tacoToUBLAS(exprOperands.at("D"),Dublas);

        TACO_BENCH(noalias(Aublas) = Bublas + Cublas + Dublas;,"UBLAS",repeat,timevalue,true);

        Tensor<double> A_ublas({rows,cols}, CSC);
        UBLASTotaco(Aublas,A_ublas);

        validate("UBLAS", A_ublas, exprOperands.at("ARef"));
        break;
      }
      case MATTRANSMUL:
      case RESIDUAL: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        UBlasSparse Aublas(rows,cols);
        tacoToUBLAS(exprOperands.at("A"),Aublas);

        UBlasDenseVector xublas(cols), zublas(rows), yublas(rows), tmpublas(rows);
        tacoToUBLAS(exprOperands.at("x"),xublas);
        tacoToUBLAS(exprOperands.at("z"),zublas);
        double alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        double beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];

        if (Expr==MATTRANSMUL) {
          TACO_BENCH(boost::numeric::ublas::axpy_prod(xublas, Aublas, tmpublas, true); yublas = alpha * tmpublas + beta * zublas;,"UBLAS",repeat,timevalue,true); }
        else {
          TACO_BENCH(boost::numeric::ublas::axpy_prod(Aublas, xublas, tmpublas, true); yublas = zublas - tmpublas ;,"UBLAS",repeat,timevalue,true); }

        Tensor<double> y_ublas({rows}, Dense);
        UBLASTotaco(yublas,y_ublas);

        validate("UBLAS", y_ublas, exprOperands.at("yRef"));
        break;
      }
      case SDDMM: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        UBlasSparse Aublas(rows,cols);
        UBlasSparse Bublas(rows,cols);
        UBlasRowMajor Cublas;
        UBlasColMajor Dublas;

        tacoToUBLAS(exprOperands.at("B"),Bublas);
        tacoToUBLAS(exprOperands.at("C"),Cublas);
        tacoToUBLAS(exprOperands.at("D"),Dublas);

        TACO_BENCH(noalias(Aublas) = element_prod(Bublas, prod(Cublas, Dublas)) ;,"UBLAS",repeat,timevalue,true);

        Tensor<double> A_ublas({rows,cols}, CSC);
        UBLASTotaco(Aublas,A_ublas);

        validate("UBLAS", A_ublas, exprOperands.at("ARef"));
        break;
      }
      default:
        cout << " !! Expression not implemented for UBLAS" << endl;
        break;
    }
}
