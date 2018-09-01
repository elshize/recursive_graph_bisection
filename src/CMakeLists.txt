add_executable(bisect main.cpp)
target_link_libraries(bisect
  rgbp
  pthread
  tbb
)
target_compile_options(bisect PRIVATE -fopenmp-simd)
