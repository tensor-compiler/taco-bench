#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "taco.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"
#include "taco/util/collections.h"
#include "taco/util/fill.h"

using namespace taco;
using namespace std;

#define TACO_BENCH(CODE, NAME, REPEAT, TIMER, COLD) {           \
    TACO_TIME_REPEAT(CODE, REPEAT, TIMER, COLD);                \
    cout << NAME << " time (ms)" << endl << TIMER << endl;      \
}

static void printFlag(string flag, string text) {
  const size_t descriptionStart = 30;
  const size_t columnEnd        = 80;
  string flagString = "  -" + flag +
      util::repeat(" ",descriptionStart-(flag.size()+3));
  cout << flagString;
  size_t column = flagString.size();
  vector<string> words = util::split(text, " ");
  for (auto& word : words) {
    if (column + word.size()+1 >= columnEnd) {
      cout << endl << util::repeat(" ", descriptionStart);
      column = descriptionStart;
    }
    column += word.size()+1;
    cout << word << " ";
  }
  cout << endl;
}

static void printUsageInfo() {
  cout << "Usage: taco <index expression> [options]" << endl;
}

static int reportError(string errorMessage, int errorCode) {
  cerr << "Error: " << errorMessage << endl << endl;
  printUsageInfo();
  return errorCode;
}

static void printCommandLine(ostream& os, int argc, char* argv[]) {
  taco_iassert(argc > 0);
  os << argv[0];
  if (argc > 1) {
    os << " \"" << argv[1] << "\"";
  }
  for (int i = 2; i < argc; i++) {
    os << " " << argv[i];
  }
}

static void readMatrixSize(string filename, int& rows, int& cols)
{
  fstream file;
  string line;
  file.open(filename);
  std::string comment("%");
  int size;

  while (getline(file,line))
  {
    if (line.compare(0, comment.size(), comment)) {
      std::istringstream iss(line);
      if (!(iss >> rows >> cols >> size)) { break; }
      break;
    }
  }
  file.close();
}

#ifdef EIGEN
#include <Eigen/Sparse>
typedef Eigen::Matrix<double,Eigen::Dynamic,1> DenseVector;
typedef Eigen::SparseMatrix<double> EigenSparseMatrix;
#endif

#ifdef GMM
#include "gmm/gmm.h"
typedef gmm::csc_matrix<double> GmmSparse;
typedef gmm::col_matrix< gmm::wsvector<double> > GmmDynSparse;
#endif

#ifdef UBLAS
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>
typedef boost::numeric::ublas::compressed_matrix<double,boost::numeric::ublas::column_major> UBlasSparse;
#endif

#ifdef OSKI
extern "C" {
#include <oski/oski.h>
}
#endif

enum BenchExpr {SpMV};

void Validate (string name, const Tensor<double>& Dst, const Tensor<double>& Ref) {
  if (!equals (Dst, Ref)) {
    cout << "\033[1;31m  Validation Error with " << name << " \033[0m" << endl;
  }
}

int main(int argc, char* argv[]) {

  int Expression=1;
  BenchExpr Expr;
  int repeat=1;
  map<string,string> inputFilenames;
  taco::util::TimeResults timevalue;

  // Read Parameters
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    vector<string> argparts = util::split(arg, "=");
    if (argparts.size() > 2) {
      return reportError("Too many '\"' signs in argument", 5);
    }
    string argName = argparts[0];
    string argValue;
    if (argparts.size() == 2)
      argValue = argparts[1];

    if ("-E" == argName) {
      try {
        Expression=stoi(argValue);
        if (Expression==1)
          Expr=SpMV;
      }
      catch (...) {
        return reportError("Expression descriptor", 3);
      }
    }
    else if ("-i" == argName) {
      vector<string> descriptor = util::split(argValue, ":");
      if (descriptor.size() != 2) {
        return reportError("Incorrect -i usage", 3);
      }
      string tensorName = descriptor[0];
      string fileName  = descriptor[1];
      inputFilenames.insert({tensorName,fileName});
    }
    if ("-r" == argName) {
      try {
        repeat=stoi(argValue);
      }
      catch (...) {
        return reportError("repeat descriptor", 3);
      }
    }
  }

  // taco expression
  map<string,Format> TacoFormats;
  TacoFormats.insert({"CSR",CSR});
  TacoFormats.insert({"Sparse,Sparse",Format({Sparse,Sparse})});
  TacoFormats.insert({"CSC",CSC});

  switch(Expr) {
    case SpMV: {
      int rows,cols;
      readMatrixSize(inputFilenames.at("A"),rows,cols);
      Tensor<double> x({cols}, Dense);
      util::fillTensor(x,util::FillMethod::Dense);
      Tensor<double> yRef({rows}, Dense);
      Tensor<double> A=read(inputFilenames.at("A"),CSC,true);
      IndexVar i, j;
      yRef(i) = A(i,j) * x(j);
      yRef.compile();
      yRef.assemble();
      yRef.compute();

      for (auto& formats:TacoFormats) {
        cout << "y(i) = A(i,j)*x(j) -- " << formats.first <<endl;
        A=read(inputFilenames.at("A"),formats.second,true);
        Tensor<double> y({rows}, Dense);

        y(i) = A(i,j) * x(j);

        TACO_BENCH(y.compile();, "Compile",1,timevalue,false)
        TACO_BENCH(y.assemble();,"Assemble",1,timevalue,false)
        TACO_BENCH(y.compute();, "Compute",repeat, timevalue, true)

        Validate("taco", y, yRef);
      }
      A=read(inputFilenames.at("A"),CSC,true);

#ifdef EIGEN
      DenseVector xEigen(cols);
      DenseVector yEigen(rows);
      EigenSparseMatrix AEigen(rows,cols);

      int r=0;
      for (auto& value : iterate<double>(x)) {
        xEigen(r++) = value.second;
      }

      std::vector< Eigen::Triplet<double> > tripletList;
      tripletList.reserve(A.getStorage().getValues().getSize());
      for (auto& value : iterate<double>(A)) {
        tripletList.push_back({value.first.at(0),value.first.at(1),value.second});
      }
      AEigen.setFromTriplets(tripletList.begin(), tripletList.end());

      TACO_BENCH(yEigen.noalias() = AEigen * xEigen;,"Eigen",repeat,timevalue,true);

      Tensor<double> y_Eigen({rows}, Dense);
      for (int i=0; i<rows; ++i) {
        y_Eigen.insert({i}, yEigen(i));
      }
      y_Eigen.pack();

      Validate("Eigen", y_Eigen, yRef);
#endif

#ifdef GMM
      GmmSparse Agmm(rows,cols);

      GmmDynSparse tmp(rows, cols);
      for (auto& value : iterate<double>(A)) {
        tmp(value.first.at(0),value.first.at(1)) = value.second;
      }
      gmm::copy(tmp, Agmm);

      std::vector<double> xgmm(cols), ygmm(rows);
      int s=0;
      for (auto& value : iterate<double>(x)) {
        xgmm[s++] = value.second;
      }

      TACO_BENCH(gmm::mult(Agmm, xgmm, ygmm);,"GMM",repeat,timevalue,true);

      Tensor<double> y_gmm({rows}, Dense);
      for (int i=0; i<rows; ++i) {
        y_gmm.insert({i}, ygmm[i]);
      }
      y_gmm.pack();

      Validate("GMM++", y_gmm, yRef);
#endif

#ifdef UBLAS
      UBlasSparse Aublas(rows,cols);

      for (auto& value : iterate<double>(A)) {
        Aublas(value.first.at(0),value.first.at(1)) = value.second;
      }

      boost::numeric::ublas::vector<double> xublas(cols), yublas(rows);
      int t=0;
      for (auto& value : iterate<double>(x)) {
        xublas[t++] = value.second;
      }

      TACO_BENCH(boost::numeric::ublas::axpy_prod(Aublas, xublas, yublas, true);,"UBLAS",repeat,timevalue,true);

      Tensor<double> y_ublas({rows}, Dense);
      for (int i=0; i<rows; ++i) {
        y_ublas.insert({i}, yublas[i]);
      }
      y_ublas.pack();

      Validate("UBLAS", y_ublas, yRef);
#endif

#ifdef OSKI
      oski_matrix_t Aoski;
      oski_vecview_t xoski, yoski;
      oski_Init();
      // TODO use getCSCArrays method
      //      int** colptr;
      //      int** rowidx;
      //      double** vals;
      //      getCSCArrays(A,colptr,rowidx,vals);
      Aoski = oski_CreateMatCSC((int*)(A.getStorage().getIndex().getDimensionIndex(1).getIndexArray(0).getData()),
                                (int*)(A.getStorage().getIndex().getDimensionIndex(1).getIndexArray(1).getData()),
                                (double*)(A.getStorage().getValues().getData()),
                                rows, cols,
                                SHARE_INPUTMAT, 1, INDEX_ZERO_BASED);
      xoski = oski_CreateVecView((double*)(x.getStorage().getValues().getData()), cols, STRIDE_UNIT);
      Tensor<double> y_oski({rows}, Dense);
      y_oski.pack();
      yoski = oski_CreateVecView((double*)(y_oski.getStorage().getValues().getData()), rows, STRIDE_UNIT);

      TACO_BENCH( oski_MatMult(Aoski, OP_NORMAL, 1, xoski, 0, yoski);,"OSKI",repeat,timevalue,true );

      Validate("OSKI", y_oski, yRef);

      // Tuned version
      oski_SetHintMatMult(Aoski, OP_NORMAL, 1.0, SYMBOLIC_VEC, 0.0, SYMBOLIC_VEC, ALWAYS_TUNE_AGGRESSIVELY);
      oski_TuneMat(Aoski);
      char* xform = oski_GetMatTransforms (Aoski);
      int blockSize=0;
      if (xform) {
        fprintf (stdout, "\tDid tune: '%s'\n", xform);
        std::string oskiTune(xform);
        std::string oskiBegin=oskiTune.substr(oskiTune.find(",")+2);
        std::string oskiXSize=oskiBegin.substr(0,oskiBegin.find(","));
        int XOski=atoi(oskiXSize.c_str());
        if (XOski!=0)
          blockSize = XOski;
        oski_Free (xform);
      }

      TACO_BENCH(oski_MatMult(Aoski, OP_NORMAL, 1, xoski, 0, yoski);,"OSKI Tuned",repeat,timevalue,true);

      Validate("OSKI Tuned", y_oski, yRef);

      oski_DestroyMat(Aoski);
      oski_DestroyVecView(xoski);
      oski_DestroyVecView(yoski);
      oski_Close();

      // Taco block version with oski tuned number
      if (blockSize>0) {
        cout << "y(i,ib) = A(i,j,ib,jb)*x(j,jb) -- DSDD " <<endl;

        IndexVar ib,jb;
        Tensor<double> yb({rows/blockSize,blockSize}, Format({Dense,Dense}));
        Tensor<double> xb({cols/blockSize,blockSize}, Format({Dense,Dense}));
        Tensor<double> Ab({rows/blockSize,cols/blockSize,blockSize,blockSize}, Format({Dense,Sparse,Dense,Dense}));

        int i_b=0;
        for (auto& value : iterate<double>(x)) {
          xb.insert({value.first.at(0)/blockSize,value.first.at(0)%blockSize},value.second);
          i_b++;
        }
        xb.pack();
        for (auto& value : iterate<double>(A)) {
          Ab.insert({value.first.at(0)/blockSize,value.first.at(1)/blockSize,value.first.at(0)%blockSize,value.first.at(1)%blockSize}, value.second);
        }
        Ab.pack();

        yb(i,ib) = Ab(i,j,ib,jb) * xb(j,jb);

        TACO_BENCH(yb.compile();, "Compile",1,timevalue,false)
        TACO_BENCH(yb.assemble();,"Assemble",1,timevalue,false)
        TACO_BENCH(yb.compute();, "Compute",repeat, timevalue, true)
      }
#endif

#ifdef POSKI
#endif


#ifdef MKL
#endif


      break;
    }
    default: {
      return reportError("Unknown Expression", 3);
    }
  }
}

