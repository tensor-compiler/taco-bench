#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef EIGEN
#include <Eigen/Sparse>
typedef Eigen::Matrix<double,Eigen::Dynamic,1> DenseVector;
typedef Eigen::SparseMatrix<double> EigenSparseMatrix;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DenseMatrix;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> DenseRowMajorMatrix;

  void EigenTotaco(const EigenSparseMatrix& src, Tensor<double>& dst)
  {
    for (int j=0; j<src.cols(); ++j)
      for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
        dst.insert({it.index(),j}, it.value());
    dst.pack();
  }

  void tacoToEigen(const Tensor<double>& src, EigenSparseMatrix& dst){
    taco_uassert(src.getFormat()==CSC)<<"Tensor have to be in CSC format to be converted to Eigen";
    std::vector< Eigen::Triplet<double> > tripletList;
    tripletList.reserve(src.getStorage().getValues().getSize());
    for (auto& value : iterate<double>(src)) {
      tripletList.push_back({value.first.at(0),value.first.at(1),value.second});
    }
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
  }

  void tacoToEigen(const Tensor<double>& src, DenseMatrix& dst){
    for (auto& value : iterate<double>(src)) {
      dst(value.first.at(0),value.first.at(1)) = value.second;
    }
  }

  void tacoToEigen(const Tensor<double>& src, DenseRowMajorMatrix& dst){
    for (auto& value : iterate<double>(src)) {
      dst(value.first.at(0),value.first.at(1)) = value.second;
    }
  }

  void EigenTotaco(const DenseVector& src, Tensor<double>& dst)  {
    for (int j=0; j<src.rows(); ++j)
      dst.insert({j}, src(j));
    dst.pack();
  }

  void tacoToEigen(const Tensor<double>& src, DenseVector& dst)  {
    for (auto& value : iterate<double>(src))
      dst(value.first[0]) = value.second;
  }

  void exprToEIGEN(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        DenseVector xEigen(cols);
        DenseVector yEigen(rows);
        EigenSparseMatrix AEigen(rows,cols);

        tacoToEigen(exprOperands.at("x"),xEigen);
        tacoToEigen(exprOperands.at("A"),AEigen);

        TACO_BENCH(yEigen.noalias() = AEigen * xEigen;,"Eigen",repeat,timevalue,true);

        Tensor<double> y_Eigen({rows}, Dense);
        EigenTotaco(yEigen,y_Eigen);

        validate("Eigen", y_Eigen, exprOperands.at("yRef"));
        break;
      }
      case PLUS3: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        EigenSparseMatrix AEigen(rows,cols);
        EigenSparseMatrix BEigen(rows,cols);
        EigenSparseMatrix CEigen(rows,cols);
        EigenSparseMatrix DEigen(rows,cols);

        tacoToEigen(exprOperands.at("B"),BEigen);
        tacoToEigen(exprOperands.at("C"),CEigen);
        tacoToEigen(exprOperands.at("D"),DEigen);

        TACO_BENCH(AEigen = BEigen + CEigen + DEigen;,"Eigen",repeat,timevalue,true);

        Tensor<double> A_Eigen({rows,cols}, CSC);
        EigenTotaco(AEigen,A_Eigen);

        validate("Eigen", A_Eigen, exprOperands.at("ARef"));
        break;
      }
      case MATTRANSMUL:
      case RESIDUAL: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        DenseVector xEigen(cols);
        DenseVector zEigen(rows);
        DenseVector yEigen(rows);
        double alpha, beta;
        EigenSparseMatrix AEigen(rows,cols);

        tacoToEigen(exprOperands.at("x"),xEigen);
        tacoToEigen(exprOperands.at("z"),zEigen);
        tacoToEigen(exprOperands.at("A"),AEigen);
        alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];

        if (Expr==MATTRANSMUL) {
          TACO_BENCH(yEigen.noalias() = alpha *AEigen.transpose() * xEigen + beta * zEigen;,"Eigen",repeat,timevalue,true);}
        else {
          TACO_BENCH(yEigen.noalias() = zEigen - AEigen * xEigen ;,"Eigen",repeat,timevalue,true); }

        Tensor<double> y_Eigen({rows}, Dense);
        EigenTotaco(yEigen,y_Eigen);

        validate("Eigen", y_Eigen, exprOperands.at("yRef"));
        break;
      }
      case SDDMM: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        EigenSparseMatrix AEigen(rows,cols);
        EigenSparseMatrix BEigen(rows,cols);
        DenseRowMajorMatrix CEigen(rows,100);
        DenseMatrix DEigen(100,cols);

        tacoToEigen(exprOperands.at("B"),BEigen);
        tacoToEigen(exprOperands.at("C"),CEigen);
        tacoToEigen(exprOperands.at("D"),DEigen);

        TACO_BENCH(AEigen = BEigen.cwiseProduct(CEigen.lazyProduct(DEigen));,"Eigen",repeat,timevalue,true);

        Tensor<double> A_Eigen({rows,cols}, CSC);
        EigenTotaco(AEigen,A_Eigen);

        validate("Eigen", A_Eigen, exprOperands.at("ARef"));
        break;
      }
      default:
        cout << " !! Expression not implemented for Eigen" << endl;
        break;
    }
  }

#endif
