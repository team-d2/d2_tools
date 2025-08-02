cmake_minimum_required(VERSION 3.8)
project(d2_tools)

# find package
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# compiler settings & policies
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

  if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_definitions(NDEBUG)
  endif()

  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# OpenMP support
find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  message(STATUS "OpenMP found and enabled.")
else()
  message(WARNING "OpenMP not found. Parallel processing will be disabled.")
endif()

# find external packages
find_package(Sophus REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration)
find_package(Boost REQUIRED COMPONENTS date_time)
find_package(small_gicp REQUIRED)

# ROS IDL generation
set(msg_files
  "msg/Diagnostics.msg"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs
)

# -----------------------------------------------------------
file(GLOB_RECURSE localizer_node_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/ros2/localizer_node.cpp"
)
list(LENGTH localizer_node_srcs localizer_node_srcs_length)
ament_auto_add_library(localizer_node SHARED
  ${localizer_node_srcs}
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(localizer_node ${cpp_typesupport_target})

target_include_directories(localizer_node PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${small_gicp_INCLUDE_DIRS}
  ${ndt_omp_INCLUDE_DIRS})
  
target_link_libraries(localizer_node
  ${PCL_LIBRARIES}
  Boost::date_time
  Eigen3::Eigen
  ${ndt_omp_LIBRARIES}
  ${small_gicp_LIBRARIES}
  Sophus::Sophus
  OpenMP::OpenMP_CXX)

rclcpp_components_register_nodes(localizer_node
  "d2::tools::ros2::LocalizerNode"
)

# -----------------------------------------------------------
ament_auto_add_library(imu_transform_component SHARED
  src/imu_transform_component.cpp
)
target_link_libraries(imu_transform_component
  Eigen3::Eigen
)
rclcpp_components_register_nodes(imu_transform_component
  "d2_tools::IMUTransformComponent"
)

foreach(target localizer_node imu_transform_component)
  if(TARGET ${target})
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()

install(TARGETS localizer_node imu_transform_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
# -----------------------------------------------------------

install(DIRECTORY 
  config
  launch
  rviz
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  ament_auto_find_test_dependencies()
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
