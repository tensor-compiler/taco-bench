#include "taco/tensor.h"

using namespace taco;
using namespace std;

#ifdef POSKI
extern "C" {
#include <poski/poski.h>
}

  void tacoToPOSKI(const Tensor<double>& src, poski_mat_t& dst) {
    int rows=src.getDimension(0);
    int cols=src.getDimension(1);
    // convert to CSR
    Tensor<double> ACSR({rows,cols}, CSR);
    for (auto& value : iterate<double>(src)) {
      ACSR.insert({value.first.at(0),value.first.at(1)},value.second);
    }
    ACSR.pack();
    double *a_CSR;
    int* ia_CSR;
    int* ja_CSR;
    getCSRArrays(ACSR,&ia_CSR,&ja_CSR,&a_CSR);

    // default thread object
    poski_threadarg_t *poski_thread = poski_InitThreads();
    poski_ThreadHints(poski_thread, NULL, POSKI_OPENMP, 12);
    poski_partitionarg_t *mat_partition = NULL;

    // create CSR matrix
     dst = poski_CreateMatCSR(ia_CSR, ja_CSR, a_CSR,
                              rows, cols, ACSR.getStorage().getValues().getSize(),
                              COPY_INPUTMAT,  // greatest flexibility in tuning
                              poski_thread, mat_partition, 2, INDEX_ZERO_BASED, MAT_GENERAL);
  }

  void tacoToPOSKI(const Tensor<double>& src, poski_vec_t& dst) {
    int cols=src.getDimension(0);
    dst = poski_CreateVec((double*)(src.getStorage().getValues().getData()),
                          cols, STRIDE_UNIT, NULL);
  }

  void exprToPOSKI(BenchExpr Expr, map<string,Tensor<double>> exprOperands,int repeat, taco::util::TimeResults timevalue) {
    switch(Expr) {
      case SpMV: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);

        // Adding extras to let POSKI tune
        int extra = 0;
        if (rows%8)
          extra = 8-rows%8;
        Tensor<double> xposki({cols+extra}, Dense);
        for (auto& value : iterate<double>(exprOperands.at("x"))) {
          xposki.insert({value.first[0]}, value.second);
        }
        xposki.pack();

        poski_Init();

        poski_mat_t A_tunable;
        tacoToPOSKI(exprOperands.at("A"),A_tunable);
        Tensor<double> y_poski({rows}, Dense);
        y_poski.pack();
        poski_vec_t xposki_view, yposki_view;
        tacoToPOSKI(y_poski,yposki_view);
        xposki_view = poski_CreateVec((double*)(xposki.getStorage().getValues().getData()), cols, STRIDE_UNIT, NULL);

        TACO_BENCH(poski_MatMult(A_tunable, OP_NORMAL, 1, xposki_view, 0, yposki_view);,"\nPOSKI",repeat,timevalue,true)

        validate("POSKI", y_poski, exprOperands.at("yRef"));

        // tune
        poski_TuneHint_MatMult(A_tunable, OP_NORMAL, 1, xposki_view, 0, yposki_view, ALWAYS_TUNE_AGGRESSIVELY);
        poski_TuneMat(A_tunable);

        TACO_BENCH(poski_MatMult(A_tunable, OP_NORMAL, 1, xposki_view, 0, yposki_view);,"\nPOSKI Tuned",repeat,timevalue,true);

        validate("POSKI Tuned", y_poski, exprOperands.at("yRef"));

        // deallocate everything -- commented because of some crashes
    //    poski_DestroyMat(A_tunable);
    //    poski_DestroyVec(xoski_view);
    //    poski_DestroyVec(yoski_view);
    //    poski_DestroyThreads(poski_thread);
    //    poski_Close();
        break;
      }
      case MATTRANSMUL:
      case RESIDUAL: {
        int rows=exprOperands.at("A").getDimension(0);
        int cols=exprOperands.at("A").getDimension(1);

        int extra = 0;
        if (rows%8)
          extra = 8-rows%8;
        Tensor<double> xposki({cols+extra}, Dense);
        for (auto& value : iterate<double>(exprOperands.at("x"))) {
          xposki.insert({value.first[0]}, value.second);
        }
        xposki.pack();

        poski_Init();

        poski_mat_t A_tunable;
        tacoToPOSKI(exprOperands.at("A"),A_tunable);
        Tensor<double> y_poski({rows}, Dense);
        y_poski.pack();
        poski_vec_t xposki_view, yposki_view, zposki_view;
        tacoToPOSKI(y_poski,yposki_view);
        tacoToPOSKI(exprOperands.at("z"),zposki_view);
        xposki_view = poski_CreateVec((double*)(xposki.getStorage().getValues().getData()), cols, STRIDE_UNIT, NULL);
        double alpha = ((double*)(exprOperands.at("alpha").getStorage().getValues().getData()))[0];
        double beta = ((double*)(exprOperands.at("beta").getStorage().getValues().getData()))[0];

        double* yvals=((double*)(y_poski.getStorage().getValues().getData()));
        double* zvals=((double*)(exprOperands.at("z").getStorage().getValues().getData()));

        if (Expr==MATTRANSMUL) {
          TACO_BENCH(for (auto k=0; k<rows; k++) {yvals[k]=zvals[k];} ;
                     poski_MatMult(A_tunable, OP_TRANS, alpha, xposki_view, beta, yposki_view);,"\nPOSKI",repeat,timevalue,true) }
        else {
          TACO_BENCH(for (auto k=0; k<rows; k++) {yvals[k]=zvals[k];} ;
                     poski_MatMult(A_tunable, OP_NORMAL, -1.0, xposki_view, 1.0, yposki_view);,"\nPOSKI",repeat,timevalue,true) }

        validate("POSKI", y_poski, exprOperands.at("yRef"));

        // tune
        if (Expr==MATTRANSMUL) {
          poski_TuneHint_MatMult(A_tunable, OP_TRANS, alpha, xposki_view, beta, yposki_view, ALWAYS_TUNE_AGGRESSIVELY); }
        else {
          poski_TuneHint_MatMult(A_tunable, OP_NORMAL, -1.0, xposki_view, 1.0, yposki_view, ALWAYS_TUNE_AGGRESSIVELY); }
        poski_TuneMat(A_tunable);

        if (Expr==MATTRANSMUL) {
          TACO_BENCH(for (auto k=0; k<rows; k++) {yvals[k]=zvals[k];} ;
                     poski_MatMult(A_tunable, OP_TRANS, alpha, xposki_view, beta, yposki_view);,"\nPOSKI Tuned",repeat,timevalue,true); }
        else {
          TACO_BENCH(for (auto k=0; k<rows; k++) {yvals[k]=zvals[k];} ;
                      poski_MatMult(A_tunable, OP_NORMAL, -1.0, xposki_view, 1.0, yposki_view);,"\nPOSKI Tuned",repeat,timevalue,true) }

        validate("POSKI Tuned", y_poski, exprOperands.at("yRef"));

        // deallocate everything -- commented because of some crashes
    //    poski_DestroyMat(A_tunable);
    //    poski_DestroyVec(xoski_view);
    //    poski_DestroyVec(yoski_view);
    //    poski_DestroyThreads(poski_thread);
    //    poski_Close();
        break;
      }
      default:
        cout << " !! Expression not implemented for POSKI" << endl;
        break;
    }
  }

#endif
