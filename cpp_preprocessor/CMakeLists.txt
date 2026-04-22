cmake_minimum_required(VERSION 3.10)
project(cpp_preprocessor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(preprocessor
    calc_amplitude.cpp
    main.cpp
    median_filter.cpp
    output_to_file.cpp
    parsing_input.cpp
    sync.cpp
)

add_executable(csi_data_generator
    generate_data.cpp
)