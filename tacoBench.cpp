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

#ifdef MKL
#endif

#ifdef OSKI
#endif

#ifdef POSKI
#endif


      break;
    }
    default: {
      return reportError("Unknown Expression", 3);
    }
  }
}

