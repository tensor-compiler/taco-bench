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
#endif

int main(int argc, char* argv[]) {

  int Expression=1;
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
  TacoFormats.insert({"CSC",CSC});
  TacoFormats.insert({"Sparse,Sparse",Format({Sparse,Sparse})});
  int rows,cols;
  readMatrixSize(inputFilenames.at("A"),rows,cols);

  switch(Expression) {
    case 1: {
      Tensor<double> x({cols}, Dense);
      util::fillTensor(x,util::FillMethod::Dense);
      for (auto& formats:TacoFormats) {
        cout << "y(i) = A(i,j)*x(j) -- " << formats.first <<endl;
        IndexVar i, j;
        Tensor<double> A=read(inputFilenames.at("A"),formats.second,true);
        Tensor<double> y({A.getDimension(0)}, Dense);

        y(i) = A(i,j) * x(j);

        TACO_BENCH(y.compile();, "Compile",1,timevalue,false)
        TACO_BENCH(y.assemble();,"Assemble",1,timevalue,false)
        TACO_BENCH(y.compute();, "Compute",repeat, timevalue, true)
      }
      break;
    }
    default: {
      return reportError("Unknown Expression", 3);
    }
  }

  // Eigen
#ifdef EIGEN
  typedef Eigen::Matrix<double,Eigen::Dynamic,1> DenseVector;
  DenseVector xEigen(cols);
//  xEigen.set
  //  yEigen = eigenBench.expression1(A,x);
  //  taco::equals(yEigen,yTaco);
#endif

}

