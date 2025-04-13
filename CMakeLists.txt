cmake_minimum_required(VERSION 3.14)

project(Perceptron LANGUAGES CXX)

add_executable(Perceptron src/main.cpp src/Network.cpp)
target_include_directories(Perceptron PRIVATE src)