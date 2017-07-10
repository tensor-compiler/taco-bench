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

// MACRO to benchmark some CODE with REPEAT times and COLD/WARM cache
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
  cout << "Usage: tacoBench [options]" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  tacoBench -E=1 -r=10 -p=eigen,ublas  -i=A:/tmp/consph.mtx" << endl;
  cout << "Options:" << endl;
  printFlag("E=<expressionId>",
            "Specify the expression Id to benchmark from: "
            "   1 : SpMV y(i)=A(i,j)*x(j) ");
  cout << endl;
  printFlag("r=<repeat>",
            "Time compilation, assembly and <repeat> times computation "
            "(defaults to 1).");
  cout << endl;
  printFlag("i=<tensor>:<filename>",
            "Read a tensor from a .mtx file.");
  cout << endl;
  printFlag("p=<product>,<products>",
            "Specify a list of products to use from: "
            "eigen, gmm, ublas, oski, poski, mkl."
            "(not specified launches all products");
  cout << endl;
}

static int reportError(string errorMessage, int errorCode) {
  cerr << "Error: " << errorMessage << endl << endl;
  printUsageInfo();
  return errorCode;
}

// Get MatrixSize from a .mtx file
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

// Compare two tensors
bool compare(const Tensor<double>&Dst, const Tensor<double>&Ref) {
  if (Dst.getDimensions() != Ref.getDimensions()) {
    return false;
  }

  {
    std::set<std::vector<int>> coords;
    for (const auto& val : Dst) {
      if (!coords.insert(val.first).second) {
        return false;
      }
    }
  }

  vector<std::pair<std::vector<int>,double>> vals;
  for (const auto& val : Dst) {
    if (val.second != 0) {
      vals.push_back(val);
    }
  }

  vector<std::pair<std::vector<int>,double>> expected;
  for (const auto& val : Ref) {
    if (val.second != 0) {
      expected.push_back(val);
    }
  }
  std::sort(expected.begin(), expected.end());
  std::sort(vals.begin(), vals.end());
  return vals == expected;
}

void validate (string name, const Tensor<double>& Dst, const Tensor<double>& Ref) {
  if (Dst.getFormat()==Ref.getFormat()) {
    if (!equals (Dst, Ref)) {
      cout << "\033[1;31m  Validation Error with " << name << " \033[0m" << endl;
    }
  }
  else {
    if (!compare(Dst,Ref)) {
      cout << "\033[1;31m  Validation Error with " << name << " \033[0m" << endl;
    }
  }
}

// Includes for all the products
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

#ifdef POSKI
extern "C" {
#include <poski/poski.h>
}
#endif

#ifdef MKL
  #include "mkl_spblas.h"
#endif

// Enum of possible expressions to Benchmark
enum BenchExpr {SpMV,plus3};

int main(int argc, char* argv[]) {

  int Expression=1;
  BenchExpr Expr;
  int repeat=1;
  map<string,string> inputFilenames;
  taco::util::TimeResults timevalue;
  map<string,bool> products;
  products.insert({"eigen",true});
  products.insert({"gmm",true});
  products.insert({"ublas",true});
  products.insert({"oski",true});
  products.insert({"poski",true});
  products.insert({"mkl",true});

  if (argc < 2)
    return reportError("no arguments", 3);

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
        else if(Expression==2)
          Expr=plus3;
        else
          return reportError("Incorrect Expression descriptor", 3);
      }
      catch (...) {
        return reportError("Incorrect Expression descriptor", 3);
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
    else if ("-p" == argName) {
      vector<string> descriptor = util::split(argValue, ",");
      if (descriptor.size() > products.size()) {
        return reportError("Incorrect -p usage", 3);
      }
      for (auto &product : products ) {
        product.second=false;
      }
      for(int i=0; i<descriptor.size();i++) {
        products.at(descriptor[i])=true;
      }
    }
    if ("-r" == argName) {
      try {
        repeat=stoi(argValue);
      }
      catch (...) {
        return reportError("Incorrect repeat descriptor", 3);
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

        validate("taco", y, yRef);
      }
      // get CSC arrays for other products
      A=read(inputFilenames.at("A"),CSC,true);
      double *a_CSC;
      int* ia_CSC;
      int* ja_CSC;
      getCSCArrays(A,&ia_CSC,&ja_CSC,&a_CSC);

#ifdef EIGEN
  if (products.at("eigen")) {
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

      validate("Eigen", y_Eigen, yRef);
  }
#else
  if (products.at("eigen")) {
    cout << "Cannot use EIGEN" << endl;
  }
#endif

#ifdef GMM
  if (products.at("gmm")) {
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

      validate("GMM++", y_gmm, yRef);
  }
#else
  if (products.at("gmm")) {
    cout << "Cannot use GMM++" << endl;
  }
#endif

#ifdef UBLAS
  if (products.at("ublas")) {
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

      validate("UBLAS", y_ublas, yRef);
  }
#else
  if (products.at("ublas")) {
    cout << "Cannot use UBLAS" << endl;
  }
#endif

#ifdef OSKI
  if (products.at("oski")) {
      oski_matrix_t Aoski;
      oski_vecview_t xoski, yoski;
      oski_Init();
      Aoski = oski_CreateMatCSC(ia_CSC,ja_CSC,a_CSC,
                                rows, cols, SHARE_INPUTMAT, 1, INDEX_ZERO_BASED);
      xoski = oski_CreateVecView((double*)(x.getStorage().getValues().getData()),
                                 cols, STRIDE_UNIT);
      Tensor<double> y_oski({rows}, Dense);
      y_oski.pack();
      yoski = oski_CreateVecView((double*)(y_oski.getStorage().getValues().getData()),
                                 rows, STRIDE_UNIT);

      TACO_BENCH( oski_MatMult(Aoski, OP_NORMAL, 1, xoski, 0, yoski);,"OSKI",repeat,timevalue,true );

      validate("OSKI", y_oski, yRef);

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

      validate("OSKI Tuned", y_oski, yRef);

      // commented to avoid some crashes with poski
//      oski_DestroyMat(Aoski);
//      oski_DestroyVecView(xoski);
//      oski_DestroyVecView(yoski);
//      oski_Close();

      // Taco block version with oski tuned number
      if (blockSize>0) {
        cout << "y(i,ib) = A(i,j,ib,jb)*x(j,jb) -- DSDD " <<endl;

        IndexVar ib,jb;
        Tensor<double> yb({rows/blockSize,blockSize}, Format({Dense,Dense}));
        Tensor<double> xb({cols/blockSize,blockSize}, Format({Dense,Dense}));
        Tensor<double> Ab({rows/blockSize,cols/blockSize,blockSize,blockSize},
                          Format({Dense,Sparse,Dense,Dense}));

        int i_b=0;
        for (auto& value : iterate<double>(x)) {
          xb.insert({value.first.at(0)/blockSize,value.first.at(0)%blockSize},
                    value.second);
          i_b++;
        }
        xb.pack();
        for (auto& value : iterate<double>(A)) {
          Ab.insert({value.first.at(0)/blockSize,value.first.at(1)/blockSize,
                     value.first.at(0)%blockSize,value.first.at(1)%blockSize},
                    value.second);
        }
        Ab.pack();

        yb(i,ib) = Ab(i,j,ib,jb) * xb(j,jb);

        TACO_BENCH(yb.compile();, "Compile",1,timevalue,false)
        TACO_BENCH(yb.assemble();,"Assemble",1,timevalue,false)
        TACO_BENCH(yb.compute();, "Compute",repeat, timevalue, true)
      }
  }
#else
  if (products.at("oski")) {
    cout << "Cannot use OSKI" << endl;
  }
#endif

#ifdef POSKI
  if (products.at("poski")) {
    // convert to CSR
    Tensor<double> ACSR=read(inputFilenames.at("A"),CSR,true);
    double *a_CSR;
    int* ia_CSR;
    int* ja_CSR;
    getCSRArrays(ACSR,&ia_CSR,&ja_CSR,&a_CSR);

    int extra = 0;
    if (rows%8)
      extra = 8-rows%8;
    Tensor<double> xposki({cols+extra}, Dense);
    int u=0;
    for (auto& value : iterate<double>(x)) {
      xposki.insert({u++}, value.second);
    }
    xposki.pack();

    poski_Init();

    // default thread object
    poski_threadarg_t *poski_thread = poski_InitThreads();
    poski_ThreadHints(poski_thread, NULL, POSKI_OPENMP, 12);
    poski_partitionarg_t *mat_partition = NULL;

    // create CSR matrix
    poski_mat_t A_tunable = poski_CreateMatCSR(ia_CSR, ja_CSR, a_CSR,
         rows, cols, ACSR.getStorage().getValues().getSize(),
         COPY_INPUTMAT,  // greatest flexibility in tuning
         poski_thread, mat_partition, 2, INDEX_ZERO_BASED, MAT_GENERAL);

    Tensor<double> y_poski({rows}, Dense);
    y_poski.pack();

    poski_vec_t xposki_view = poski_CreateVec((double*)(xposki.getStorage().getValues().getData()), cols, STRIDE_UNIT, NULL);
    poski_vec_t yposki_view = poski_CreateVec((double*)(y_poski.getStorage().getValues().getData()), rows, STRIDE_UNIT, NULL);

    TACO_BENCH(poski_MatMult(A_tunable, OP_NORMAL, 1, xposki_view, 0, yposki_view);,"POSKI",repeat,timevalue,true)

    validate("POSKI", y_poski, yRef);

    // tune
    poski_TuneHint_MatMult(A_tunable, OP_NORMAL, 1, xposki_view, 0, yposki_view, ALWAYS_TUNE_AGGRESSIVELY);
    poski_TuneMat(A_tunable);

    TACO_BENCH(poski_MatMult(A_tunable, OP_NORMAL, 1, xposki_view, 0, yposki_view);,"POSKI Tuned",repeat,timevalue,true);

    validate("POSKI Tuned", y_poski, yRef);

    // deallocate everything -- commented because of some crashes
//    poski_DestroyMat(A_tunable);
//    poski_DestroyVec(xoski_view);
//    poski_DestroyVec(yoski_view);
//    poski_DestroyThreads(poski_thread);
//    poski_Close();
  }
#else
  if (products.at("poski")) {
    cout << "Cannot use POSKI" << endl;
  }
#endif

#ifdef MKL
  if (products.at("mkl")) {
    char matdescra[6] = "G  C ";
    int ptrsize = A.getStorage().getIndex().getSize();
    int* pointerB=new int[ptrsize-1];
    int* pointerE=new int[ptrsize-1];
    for (int i=0; i<ptrsize-1; i++) {
      pointerB[i]=ia_CSC[i];
      pointerE[i]=ia_CSC[i+1];
    }
    Tensor<double> y_mkl({rows}, Dense);
    y_mkl.pack();

    double malpha=1.0;
    double mbeta=0.0;
    char transa = 'N';
    TACO_BENCH(mkl_dcscmv(&transa, &rows, &cols, &malpha, matdescra, a_CSC, ja_CSC, pointerB,
                          pointerE, (double*)(x.getStorage().getValues().getData()),
                          &mbeta, (double*)(y_mkl.getStorage().getValues().getData()));,
               "MKL", repeat,timevalue,true)

    validate("MKL", y_mkl, yRef);
  }
#else
  if (products.at("mkl")) {
    cout << "Cannot use MKL" << endl;
  }
#endif

      break;
    }
    case plus3: {
      int rows,cols;
      readMatrixSize(inputFilenames.at("B"),rows,cols);
      Tensor<double> B=read(inputFilenames.at("B"),CSC,true);
      Tensor<double> C=read(inputFilenames.at("C"),CSC,true);
      Tensor<double> D=read(inputFilenames.at("D"),CSC,true);
      Tensor<double> ARef("ARef",{rows,cols},CSC);
      IndexVar i, j;
      ARef(i,j) = B(i,j) + C(i,j) + D(i,j);
      ARef.compile();
      ARef.assemble();
      ARef.compute();

      map<string,Format> TacoFormats;
      TacoFormats.insert({"CSR",CSR});
      TacoFormats.insert({"Sparse,Sparse",Format({Sparse,Sparse})});
      TacoFormats.insert({"CSC",CSC});

      for (auto& formats:TacoFormats) {
        cout << "A(i,j) = B(i,j) + C(i,j) + D(i,j) -- " << formats.first <<endl;
        B=read(inputFilenames.at("B"),formats.second,true);
        C=read(inputFilenames.at("C"),formats.second,true);
        D=read(inputFilenames.at("D"),formats.second,true);
        Tensor<double> A({rows,cols},formats.second);

        A(i,j) = B(i,j) + C(i,j) + D(i,j);

        TACO_BENCH(A.compile();, "Compile",1,timevalue,false)
        TACO_BENCH(A.assemble();,"Assemble",1,timevalue,false)
        TACO_BENCH(A.compute();, "Compute",repeat, timevalue, true)

        validate("taco", A, ARef);
      }
      // get CSC arrays for other products
      B=read(inputFilenames.at("B"),CSC,true);
      C=read(inputFilenames.at("C"),CSC,true);
      D=read(inputFilenames.at("D"),CSC,true);

#ifdef EIGEN
  if (products.at("eigen")) {
      EigenSparseMatrix AEigen(rows,cols);
      EigenSparseMatrix BEigen(rows,cols);
      EigenSparseMatrix CEigen(rows,cols);
      EigenSparseMatrix DEigen(rows,cols);

      std::vector< Eigen::Triplet<double> > tripletList;
      tripletList.reserve(B.getStorage().getValues().getSize());
      for (auto& value : iterate<double>(B)) {
        tripletList.push_back({value.first.at(0),value.first.at(1),value.second});
      }
      BEigen.setFromTriplets(tripletList.begin(), tripletList.end());
      tripletList.clear();
      tripletList.reserve(C.getStorage().getValues().getSize());
      for (auto& value : iterate<double>(C)) {
        tripletList.push_back({value.first.at(0),value.first.at(1),value.second});
      }
      CEigen.setFromTriplets(tripletList.begin(), tripletList.end());
      tripletList.clear();
      tripletList.reserve(D.getStorage().getValues().getSize());
      for (auto& value : iterate<double>(D)) {
        tripletList.push_back({value.first.at(0),value.first.at(1),value.second});
      }
      DEigen.setFromTriplets(tripletList.begin(), tripletList.end());

      TACO_BENCH(AEigen = BEigen + CEigen + DEigen;,"Eigen",repeat,timevalue,true);

      Tensor<double> A_Eigen({rows,cols}, CSC);
      for (int j=0; j<cols; ++j)
        for (EigenSparseMatrix::InnerIterator it(AEigen.derived(), j); it; ++it)
          A_Eigen.insert({it.index(),j}, it.value());
      A_Eigen.pack();

      validate("Eigen", A_Eigen, ARef);
  }
#else
  if (products.at("eigen")) {
    cout << "Cannot use EIGEN" << endl;
  }
#endif

      break;
    }
    default: {
      return reportError("Unknown Expression", 3);
    }
  }
}

