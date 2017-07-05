cmake_minimum_required(VERSION 2.8)
project(tacoBench)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_CODE})

# To find taco
if (NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
  message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
endif ()
set(TACO_INCLUDE_DIR $ENV{TACO_INCLUDE_DIR})
set(TACO_LIBRARY_DIR $ENV{TACO_LIBRARY_DIR})
find_library(taco taco ${TACO_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco})

# Include taco headers
include_directories(${TACO_INCLUDE_DIR})

# Eigen
if (NOT DEFINED ENV{EIGEN_INCLUDE_DIR})
  message(WARNING "Eigen not found and will not be used")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN -Wno-deprecated-declarations")    
  include_directories($ENV{EIGEN_INCLUDE_DIR})
endif ()

# GMM
if (NOT DEFINED ENV{GMM_INCLUDE_DIR})
  message(WARNING "GMM++ not found and will not be used")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGMM ")    
  include_directories($ENV{GMM_INCLUDE_DIR})
endif ()

# UBLAS
if (NOT DEFINED ENV{UBLAS_INCLUDE_DIR})
  message(WARNING "UBLAS not found and will not be used")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUBLAS -DNDEBUG ")    
  include_directories($ENV{UBLAS_INCLUDE_DIR})
endif ()
   
# OSKI
if ((NOT DEFINED ENV{OSKI_INCLUDE_DIR}) OR (NOT DEFINED ENV{OSKI_LIBRARY_DIR}))
  message(WARNING "OSKI not found and will not be used")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOSKI ")    
  include_directories($ENV{OSKI_INCLUDE_DIR})
  set(OSKI_LIBRARY_DIR $ENV{OSKI_LIBRARY_DIR})
  set(OSKI_INCLUDE_DIR $ENV{OSKI_INCLUDE_DIR})
  set(OSKI_MOD_COMMAND "cat $ENV{OSKI_LIBRARY_DIR}/site-modules-shared.txt | sed ':a\;N\;$!ba\;s/\\n/ /g'  ")
  execute_process(COMMAND ${OSKI_MOD_COMMAND} OUTPUT_VARIABLE OSKI_MODULES)
  target_link_libraries(${PROJECT_NAME} "-L${OSKI_LIBRARY_DIR} -loski -loskilt -loski_Tid -loski_mat_CSR_Tid -loski_mat_CSC_Tid -loski_mat_BCSR_Tid -loski_mat_MBCSR_Tid -loski_mat_GCSR_Tid -loski_mat_CB_Tid -loski_mat_VBR_Tid -loski_mat_DENSE_Tid -loski_heur_regprof_Tid -loski_heur_symmrb_Tid -loski_heur_mregblock_Tid")
# TODO try to use the module variable
#  target_link_libraries(${PROJECT_NAME} "-L${OSKI_LIBRARY_DIR} ${OSKI_MODULES}")
endif ()

   