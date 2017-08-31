 #include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef GMM
#include "gmm/gmm.h"
typedef gmm::csc_matrix<double> GmmCSC;
typedef gmm::csr_matrix<double> GmmCSR;
typedef gmm::col_matrix< gmm::wsvector<double> > GmmSparse;
typedef gmm::linalg_traits<gmm::wsvector<double>>::const_iterator GmmIterator;

  void GMMTotaco(const GmmSparse& src, Tensor<double>& dst) {
    for (int j = 0; j < gmm::mat_ncols(src); ++j) {
      typename gmm::linalg_traits<GmmSparse>::const_sub_col_type col = mat_const_col(src, j);
      GmmIterator it1 = vect_const_begin(col);
      GmmIterator ite1 = vect_const_end(col);
      while (it1 != ite1) {
        dst.insert({(int)(it1.index()),j},*it1);
        ++it1;
      }
    }
    dst.pack();
  }

  void tacoToGMM(const Tensor<double>& src, GmmSparse& dst) {
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

        GmmSparse Agmm_tmp(rows,cols);
        tacoToGMM(exprOperands.at("A"),Agmm_tmp);
        GmmCSR Agmm(rows,cols);
        gmm::copy(Agmm_tmp, Agmm);
        std::vector<double> xgmm(cols), ygmm(rows);
        tacoToGMM(exprOperands.at("x"),xgmm);

        TACO_BENCH(gmm::mult(Agmm, xgmm, ygmm);,"\nGMM",repeat,timevalue,true);

        Tensor<double> y_gmm({rows}, Dense);
        GMMTotaco(ygmm,y_gmm);

        validate("GMM++", y_gmm, exprOperands.at("yRef"));
        break;
      }
      case PLUS3: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        GmmSparse Agmm(rows,cols);
        GmmSparse Bgmm(rows,cols);
        GmmSparse Cgmm(rows,cols);
        GmmSparse Dgmm(rows,cols);

        tacoToGMM(exprOperands.at("B"),Bgmm);
        tacoToGMM(exprOperands.at("C"),Cgmm);
        tacoToGMM(exprOperands.at("D"),Dgmm);

        TACO_BENCH(Agmm=Bgmm;gmm::add(Cgmm,Agmm);gmm::add(Dgmm,Agmm);,"\nGMM",repeat,timevalue,true);

        Tensor<double> A_gmm({rows,cols}, CSC);
        GMMTotaco(Agmm,A_gmm);

        // comment out for now as Gmm++ and taco treat physical zeros differently
        // validate("GMM", A_gmm, exprOperands.at("ARef"));
        break;
      }
      case MATTRANSMUL: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);

        GmmSparse Agmm_tmp(rows,cols);
        tacoToGMM(exprOperands.at("A"),Agmm_tmp);
        GmmCSC Agmm(rows,cols);
        gmm::copy(Agmm_tmp, Agmm);
        std::vector<double> xgmm(cols), ygmm(rows), zgmm(rows);
        tacoToGMM(exprOperands.at("x"),xgmm);
        tacoToGMM(exprOperands.at("z"),zgmm);
        double alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        double beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];

        TACO_BENCH(gmm::mult(gmm::transposed(Agmm), gmm::scaled(xgmm, alpha), gmm::scaled(zgmm, beta), ygmm);,"\nGMM",repeat,timevalue,true);

        Tensor<double> y_gmm({rows}, Dense);
        GMMTotaco(ygmm,y_gmm);

        validate("GMM++", y_gmm, exprOperands.at("yRef"));
        break;
      }
      case RESIDUAL: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);

        GmmSparse Agmm_tmp(rows,cols);
        tacoToGMM(exprOperands.at("A"),Agmm_tmp);
        GmmCSR Agmm(rows,cols);
        gmm::copy(Agmm_tmp, Agmm);
        std::vector<double> xgmm(cols), ygmm(rows), zgmm(rows);
        tacoToGMM(exprOperands.at("x"),xgmm);
        tacoToGMM(exprOperands.at("z"),zgmm);
        double alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        double beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];

        TACO_BENCH(gmm::mult(Agmm, gmm::scaled(xgmm, -1.0), zgmm, ygmm);,"\nGMM",repeat,timevalue,true);

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
