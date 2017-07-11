#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef GMM
#ifdef GMM
#include "gmm/gmm.h"
typedef gmm::csc_matrix<double> GmmSparse;
typedef gmm::col_matrix< gmm::wsvector<double> > GmmDynSparse;
#endif

  void tacoToGMM(const Tensor<double>& src, GmmSparse& dst) {
    GmmDynSparse tmp(src.getDimension(0), src.getDimension(1));
    for (auto& value : iterate<double>(src))
      tmp(value.first.at(0),value.first.at(1)) = value.second;
    gmm::copy(tmp, dst);
  }

  void GMMTotaco(const std::vector<double>& src, Tensor<double>& dst){
    for (int i=0; i<dst.getDimension(0); ++i)
      dst.insert({i}, src[i]);
    dst.pack();
  }

  void tacoToGMM(const Tensor<double>& src, std::vector<double>& dst)  {
    for (auto& value : iterate<double>(src))
      dst[value.first[0]] = value.second;
  }

  void exprToGMM(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);

        GmmSparse Agmm(rows,cols);
        tacoToGMM(exprOperands.at("A"),Agmm);
        std::vector<double> xgmm(cols), ygmm(rows);
        tacoToGMM(exprOperands.at("x"),xgmm);

        TACO_BENCH(gmm::mult(Agmm, xgmm, ygmm);,"GMM",repeat,timevalue,true);

        Tensor<double> y_gmm({rows}, Dense);
        GMMTotaco(ygmm,y_gmm);

        validate("GMM++", y_gmm, exprOperands.at("yRef"));
        break;
      }
      default:
        cout << " !! Expression not implemented for GMM" << endl;
        break;
    }

  }
#endif
