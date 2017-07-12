#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef GMM
#ifdef GMM
#include "gmm/gmm.h"
typedef gmm::csc_matrix<double> GmmSparse;
typedef gmm::col_matrix< gmm::wsvector<double> > GmmDynSparse;
#endif

  void GMMTotaco(const GmmDynSparse& src, Tensor<double>& dst) {
    for (int j = 0; j < gmm::mat_ncols(src); ++j) {
      typename gmm::linalg_traits<GmmDynSparse>::const_sub_col_type col = mat_const_col(src, j);
      typename gmm::linalg_traits<gmm::wsvector<double>>::const_iterator it1 = vect_const_begin(col);
      typename gmm::linalg_traits<gmm::wsvector<double>>::const_iterator ite1 = vect_const_end(col);
      while (it1 != ite1) {
        dst.insert({(int)(it1.index()),j},*it1);
        ++it1;
      }
    }
    dst.pack();
  }

  void tacoToGMM(const Tensor<double>& src, GmmDynSparse& dst) {
    for (auto& value : iterate<double>(src))
      dst(value.first.at(0),value.first.at(1)) = value.second;
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

        GmmDynSparse Agmm_tmp(rows,cols);
        tacoToGMM(exprOperands.at("A"),Agmm_tmp);
        GmmSparse Agmm(rows,cols);
        gmm::copy(Agmm_tmp, Agmm);
        std::vector<double> xgmm(cols), ygmm(rows);
        tacoToGMM(exprOperands.at("x"),xgmm);

        TACO_BENCH(gmm::mult(Agmm, xgmm, ygmm);,"GMM",repeat,timevalue,true);

        Tensor<double> y_gmm({rows}, Dense);
        GMMTotaco(ygmm,y_gmm);

        validate("GMM++", y_gmm, exprOperands.at("yRef"));
        break;
      }
      case PLUS3: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        GmmDynSparse Agmm(rows,cols);
        GmmDynSparse Bgmm(rows,cols);
        GmmDynSparse Cgmm(rows,cols);
        GmmDynSparse Dgmm(rows,cols);

        tacoToGMM(exprOperands.at("B"),Bgmm);
        tacoToGMM(exprOperands.at("C"),Cgmm);
        tacoToGMM(exprOperands.at("D"),Dgmm);

        TACO_BENCH(Agmm=Bgmm;gmm::add(Cgmm,Agmm);gmm::add(Dgmm,Agmm);,"GMM",repeat,timevalue,true);

        Tensor<double> A_gmm({rows,cols}, CSC);
        GMMTotaco(Agmm,A_gmm);

        validate("GMM", A_gmm, exprOperands.at("ARef"));
        break;
      }
      default:
        cout << " !! Expression not implemented for GMM" << endl;
        break;
    }

  }
#endif
