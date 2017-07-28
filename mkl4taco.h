#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef MKL
  #include "mkl_spblas.h"
  #include "mkl_blas.h"
  #include "mkl.h"

  void exprToMKL(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV: {
        char matdescra[6] = "G  C ";
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        int nnz=exprOperands.at("A").getStorage().getValues().getSize();

        // convert to CSR
        Tensor<double> ACSR({rows,cols}, CSR);
        for (auto& value : iterate<double>(exprOperands.at("A"))) {
          ACSR.insert({value.first.at(0),value.first.at(1)},value.second);
        }
        ACSR.pack();
        double *a_CSR;
        int* ia_CSR;
        int* ja_CSR;
        getCSRArrays(ACSR,&ia_CSR,&ja_CSR,&a_CSR);
        for (int i = 0; i < rows+1; ++i) {
          ia_CSR[i] = ia_CSR[i] + 1;
        }
        for (int i = 0; i < nnz; ++i) {
          ja_CSR[i] = ja_CSR[i] + 1;
        }
        Tensor<double> y_mkl({rows}, Dense);
        y_mkl.pack();

        char transa = 'N';
        TACO_BENCH(mkl_dcsrgemv(&transa, &rows, a_CSR, ia_CSR, ja_CSR,
                                (double*)(exprOperands.at("x").getStorage().getValues().getData()),
                                (double*)(y_mkl.getStorage().getValues().getData()));,
                   "MKL", repeat,timevalue,true)

        validate("MKL", y_mkl, exprOperands.at("yRef"));

        break;
      }
      case PLUS3: {
        int rows=exprOperands.at("ARef").getDimension(0);
        int cols=exprOperands.at("ARef").getDimension(1);
        int nnz=exprOperands.at("B").getStorage().getValues().getSize();
        // convert to CSR
        Tensor<double> BCSR({rows,cols}, CSR);
        for (auto& value : iterate<double>(exprOperands.at("B"))) {
          BCSR.insert({value.first.at(0),value.first.at(1)},value.second);
        }
        BCSR.pack();
        double *b_CSR;
        int* ib_CSR;
        int* jb_CSR;
        getCSRArrays(BCSR,&ib_CSR,&jb_CSR,&b_CSR);
        Tensor<double> CCSR({rows,cols}, CSR);
        for (auto& value : iterate<double>(exprOperands.at("C"))) {
          CCSR.insert({value.first.at(0),value.first.at(1)},value.second);
        }
        CCSR.pack();
        double *c_CSR;
        int* ic_CSR;
        int* jc_CSR;
        getCSRArrays(CCSR,&ic_CSR,&jc_CSR,&c_CSR);
        Tensor<double> DCSR({rows,cols}, CSR);
        for (auto& value : iterate<double>(exprOperands.at("D"))) {
          DCSR.insert({value.first.at(0),value.first.at(1)},value.second);
        }
        DCSR.pack();
        double *d_CSR;
        int* id_CSR;
        int* jd_CSR;
        getCSRArrays(DCSR,&id_CSR,&jd_CSR,&d_CSR);

        char transa = 'N';
        int ptrsize = rows + 1;
        double malpha=1.0;
        double mbeta=0.0;

        for (int i = 0; i < ptrsize; ++i) {
          ib_CSR[i] = ib_CSR[i] + 1;
          ic_CSR[i] = ic_CSR[i] + 1;
          id_CSR[i] = id_CSR[i] + 1;
        }
        for (int i = 0; i < nnz; ++i) {
          jb_CSR[i] = jb_CSR[i] + 1;
          jc_CSR[i] = jc_CSR[i] + 1;
          jd_CSR[i] = jd_CSR[i] + 1;
        }
        MKL_INT request = 0;
        MKL_INT sort = 0;
        MKL_INT ret;
        double *a_CSR;
        int* ia_CSR;
        int* ja_CSR;

        TACO_BENCH( a_CSR = new double[4*nnz]; ia_CSR = new int[4*nnz]; ja_CSR = new int[4*nnz];
             mkl_dcsradd(&transa, &request, &sort, &rows, &cols, b_CSR, jb_CSR, ib_CSR, &malpha, c_CSR, jc_CSR, ic_CSR, a_CSR, ja_CSR, ia_CSR, &nnz, &ret);
             mkl_dcsradd(&transa, &request, &sort, &rows, &cols, a_CSR, ja_CSR, ia_CSR, &malpha, d_CSR, jd_CSR, id_CSR, a_CSR, ja_CSR, ia_CSR, &nnz, &ret);,
             "MKL",repeat,timevalue,true);

        for (int i = 0; i < ptrsize; ++i) {
          ia_CSR[i] = ia_CSR[i] - 1;
        }
        for (int i = 0; i < nnz; ++i) {
          ja_CSR[i] = ja_CSR[i] - 1;
        }
        Tensor<double> AMKL=makeCSR("AMKL",{rows,cols}, ia_CSR, ja_CSR, a_CSR);
        validate("MKL", AMKL, exprOperands.at("ARef"));

        break;
      }
      case MATTRANSMUL: {
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

        double alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        double beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];
        double* yvals=((double*)(y_mkl.getStorage().getValues().getData()));
        double* zvals=((double*)(exprOperands.at("z").getStorage().getValues().getData()));

        char transa = 'T';
        TACO_BENCH(for (auto k=0; k<rows; k++) {yvals[k]=zvals[k];} ;
        mkl_dcscmv(&transa, &rows, &cols, &alpha, matdescra, a_CSC, ja_CSC, pointerB,
                   pointerE, (double*)(exprOperands.at("x").getStorage().getValues().getData()),
                   &beta, (double*)(y_mkl.getStorage().getValues().getData()));,
                   "MKL", repeat,timevalue,true)

        validate("MKL", y_mkl, exprOperands.at("yRef"));

        break;
      }
      case RESIDUAL: {
        char matdescra[6] = "G  C ";
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);
        int nnz=exprOperands.at("A").getStorage().getValues().getSize();
        // convert to CSR
        Tensor<double> ACSR({rows,cols}, CSR);
        for (auto& value : iterate<double>(exprOperands.at("A"))) {
          ACSR.insert({value.first.at(0),value.first.at(1)},value.second);
        }
        ACSR.pack();
        double *a_CSR;
        int* ia_CSR;
        int* ja_CSR;
        getCSRArrays(ACSR,&ia_CSR,&ja_CSR,&a_CSR);
        double alpha=-1.0;
        double beta=1.0;

        int* pointerB=new int[nnz];
        int* pointerE=new int[nnz];
        for (int i=0; i<nnz; i++) {
          pointerB[i]=ia_CSR[i];
          pointerE[i]=ia_CSR[i+1];
        }

        Tensor<double> y_mkl({rows}, Dense);
        y_mkl.pack();

        double* yvals=((double*)(y_mkl.getStorage().getValues().getData()));
        double* zvals=((double*)(exprOperands.at("z").getStorage().getValues().getData()));

        char transa = 'N';
        TACO_BENCH(for (auto k=0; k<rows; k++) {yvals[k]=zvals[k];} ;
        mkl_dcsrmv(&transa, &rows, &cols, &alpha, matdescra, a_CSR, ja_CSR, pointerB,
                   pointerE, (double*)(exprOperands.at("x").getStorage().getValues().getData()),
                   &beta, (double*)(y_mkl.getStorage().getValues().getData()));,
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
