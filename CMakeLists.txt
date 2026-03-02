cmake_minimum_required(VERSION 3.10)
project(NeuralCharRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find SFML ───────────────────────────────────────────────
# Set SFML_DIR if cmake can't find it automatically:
#   cmake -DSFML_DIR="C:/SFML-2.5.1/lib/cmake/SFML" ..
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# ── Shared drawing library ──────────────────────────────────
add_library(draw STATIC common/draw.cpp common/draw.h)
target_include_directories(draw PUBLIC common)
target_link_libraries(draw PUBLIC sfml-graphics sfml-window sfml-system)

# ── Main selector (no SFML needed) ──────────────────────────
add_executable(neural_selector main.cpp)

# ── Algorithm executables ───────────────────────────────────
# Each one links the shared 'draw' library which brings in SFML

# Hebb
add_executable(hebb_main
    algorithms/hebb/main.cpp
    algorithms/hebb/hebb.cpp
    algorithms/hebb/hebb.h
)
target_include_directories(hebb_main PRIVATE algorithms/hebb)
target_link_libraries(hebb_main PRIVATE draw)

# Perceptron
add_executable(perceptron_main
    algorithms/perceptron/main.cpp
    algorithms/perceptron/perceptron.cpp
    algorithms/perceptron/perceptron.h
)
target_include_directories(perceptron_main PRIVATE algorithms/perceptron)
target_link_libraries(perceptron_main PRIVATE draw)

# Adaline
add_executable(adaline_main
    algorithms/adaline/main.cpp
    algorithms/adaline/adaline.cpp
    algorithms/adaline/adaline.h
)
target_include_directories(adaline_main PRIVATE algorithms/adaline)
target_link_libraries(adaline_main PRIVATE draw)

# Madaline
add_executable(madaline_main
    algorithms/madaline/main.cpp
    algorithms/madaline/madaline.cpp
    algorithms/madaline/madaline.h
)
target_include_directories(madaline_main PRIVATE algorithms/madaline)
target_link_libraries(madaline_main PRIVATE draw)

# MLP
add_executable(mlp_main
    algorithms/mlp/main.cpp
    algorithms/mlp/mlp.cpp
    algorithms/mlp/mlp.h
)
target_include_directories(mlp_main PRIVATE algorithms/mlp)
target_link_libraries(mlp_main PRIVATE draw)

# ── Put all executables in project root / bin ────────────────
set_target_properties(
    neural_selector hebb_main perceptron_main adaline_main madaline_main mlp_main
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── Copy training data to build output ──────────────────────
file(COPY data/training.txt DESTINATION "${CMAKE_BINARY_DIR}/bin/data")

# ── Copy SFML DLLs on Windows ──────────────────────────────
if(WIN32)
    # Copies SFML .dll files next to executables automatically
    foreach(target hebb_main perceptron_main adaline_main madaline_main mlp_main)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${SFML_DIR}/../../../bin"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Copying SFML DLLs for ${target}"
        )
    endforeach()
endif()
