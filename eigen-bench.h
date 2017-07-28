#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef EIGEN
#include <Eigen/Sparse>
typedef Eigen::Matrix<double,Eigen::Dynamic,1> DenseVector;
typedef Eigen::SparseMatrix<double> EigenCSC;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> EigenCSR;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> EigenColMajor;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> EigenRowMajor;

  void EigenTotaco(const EigenCSC& src, Tensor<double>& dst)
  {
    for (int j=0; j<src.cols(); ++j)
      for (EigenCSC::InnerIterator it(src.derived(), j); it; ++it)
        dst.insert({it.index(),j}, it.value());
    dst.pack();
  }

  void tacoToEigen(const Tensor<double>& src, EigenCSC& dst){
    taco_uassert(src.getFormat()==CSC)<<"Tensor have to be in CSC format to be converted to Eigen";
    std::vector< Eigen::Triplet<double> > tripletList;
    tripletList.reserve(src.getStorage().getValues().getSize());
    for (auto& value : iterate<double>(src)) {
      tripletList.push_back({value.first.at(0),value.first.at(1),value.second});
    }
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
  }

  void tacoToEigen(const Tensor<double>& src, EigenCSR& dst){
    taco_uassert(src.getFormat()==CSR)<<"Tensor have to be in CSR format to be converted to Eigen";
    std::vector< Eigen::Triplet<double> > tripletList;
    tripletList.reserve(src.getStorage().getValues().getSize());
    for (auto& value : iterate<double>(src)) {
      tripletList.push_back({value.first.at(0),value.first.at(1),value.second});
    }
    dst.setFromTriplets(tripletList.begin(), tripletList.end());
  }

  void tacoToEigen(const Tensor<double>& src, EigenColMajor& dst){
    for (auto& value : iterate<double>(src)) {
      dst(value.first.at(0),value.first.at(1)) = value.second;
    }
  }

  void tacoToEigen(const Tensor<double>& src, EigenRowMajor& dst){
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
        EigenCSR AEigen(rows,cols);

        tacoToEigen(exprOperands.at("x"),xEigen);
        tacoToEigen(exprOperands.at("A"),AEigen);

        TACO_BENCH(yEigen.noalias() = AEigen * xEigen;,"\nEigen",repeat,timevalue,true);

        Tensor<double> y_Eigen({rows}, Dense);
        EigenTotaco(yEigen,y_Eigen);

        validate("Eigen", y_Eigen, exprOperands.at("yRef"));
        break;
      }
      case PLUS3: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        EigenCSC AEigen(rows,cols);
        EigenCSC BEigen(rows,cols);
        EigenCSC CEigen(rows,cols);
        EigenCSC DEigen(rows,cols);

        tacoToEigen(exprOperands.at("B"),BEigen);
        tacoToEigen(exprOperands.at("C"),CEigen);
        tacoToEigen(exprOperands.at("D"),DEigen);

        TACO_BENCH(AEigen = BEigen + CEigen + DEigen;,"\nEigen",repeat,timevalue,true);

        Tensor<double> A_Eigen({rows,cols}, CSC);
        EigenTotaco(AEigen,A_Eigen);

        validate("Eigen", A_Eigen, exprOperands.at("ARef"));
        break;
      }
      case MATTRANSMUL: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        DenseVector xEigen(cols);
        DenseVector zEigen(rows);
        DenseVector yEigen(rows);
        double alpha, beta;
        EigenCSC AEigen(rows,cols);

        tacoToEigen(exprOperands.at("x"),xEigen);
        tacoToEigen(exprOperands.at("z"),zEigen);
        tacoToEigen(exprOperands.at("A"),AEigen);
        alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];

        TACO_BENCH(yEigen.noalias() = alpha *AEigen.transpose() * xEigen + beta * zEigen;,"\nEigen",repeat,timevalue,true);

        Tensor<double> y_Eigen({rows}, Dense);
        EigenTotaco(yEigen,y_Eigen);

        validate("Eigen", y_Eigen, exprOperands.at("yRef"));
        break;
      }
      case RESIDUAL: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        DenseVector xEigen(cols);
        DenseVector zEigen(rows);
        DenseVector yEigen(rows);
        double alpha, beta;
        EigenCSR AEigen(rows,cols);

        tacoToEigen(exprOperands.at("x"),xEigen);
        tacoToEigen(exprOperands.at("z"),zEigen);
        tacoToEigen(exprOperands.at("A"),AEigen);
        alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];

        TACO_BENCH(yEigen.noalias() = zEigen - AEigen * xEigen ;,"\nEigen",repeat,timevalue,true);

        Tensor<double> y_Eigen({rows}, Dense);
        EigenTotaco(yEigen,y_Eigen);

        validate("Eigen", y_Eigen, exprOperands.at("yRef"));
        break;
      }
      case SDDMM: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        EigenCSC AEigen(rows,cols);
        EigenCSC BEigen(rows,cols);
        EigenRowMajor CEigen(rows,100);
        EigenColMajor DEigen(100,cols);

        tacoToEigen(exprOperands.at("B"),BEigen);
        tacoToEigen(exprOperands.at("C"),CEigen);
        tacoToEigen(exprOperands.at("D"),DEigen);

        TACO_BENCH(AEigen = BEigen.cwiseProduct(CEigen.lazyProduct(DEigen));,"\nEigen",repeat,timevalue,true);

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
