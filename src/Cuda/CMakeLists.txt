# Modified to add cuda files
macro(add_cuda_source_group module_name)
  file(GLOB MODULE_HEADER_FILES "${module_name}/*.cuh")
  source_group("Header Files\\${module_name}" FILES ${MODULE_HEADER_FILES})
  file(GLOB MODULE_SOURCE_FILES "${module_name}/*.cu")
  source_group("Source Files\\${module_name}" FILES ${MODULE_SOURCE_FILES})
endmacro()

find_package(CUDAToolkit REQUIRED)

# GLOB all required files in this subdirectory
file(GLOB_RECURSE CUDA_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE CUDA_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE CUDA_ALL_CUDA_HEADER_FILES "*.cuh")
file(GLOB_RECURSE CUDA_ALL_CUDA_SOURCE_FILES "*.cu")

# create subgroups
add_source_group(Common)
add_source_group(Camera)
add_source_group(Container)
add_source_group(Geometry)
add_source_group(Integration)
add_source_group(Registration)
add_source_group(IO)
add_source_group(Odometry)
add_source_group(Visualization)
add_source_group(OSLAMUtils)

# and also for cuda
add_cuda_source_group(Common)
add_cuda_source_group(Camera)
add_cuda_source_group(Container)
add_cuda_source_group(Geometry)
add_cuda_source_group(Integration)
add_cuda_source_group(Registration)
add_cuda_source_group(IO)
add_cuda_source_group(Odometry)
add_cuda_source_group(Visualization)
add_cuda_source_group(OSLAMUtils)

# create (should be static) library
set(OPEN3D_CUDA_MODULE "Open3DCuda" CACHE STRING "CUDA component library")

add_library(
  ${OPEN3D_CUDA_MODULE}
  ${CUDA_ALL_HEADER_FILES}
  ${CUDA_ALL_SOURCE_FILES}
  ${CUDA_ALL_CUDA_HEADER_FILES}
  ${CUDA_ALL_CUDA_SOURCE_FILES}
  $<TARGET_OBJECTS:Camera>
  $<TARGET_OBJECTS:ColorMap>
  $<TARGET_OBJECTS:Geometry>
  $<TARGET_OBJECTS:Integration>
  $<TARGET_OBJECTS:Odometry>
  $<TARGET_OBJECTS:Registration>
  $<TARGET_OBJECTS:Utility>
  $<TARGET_OBJECTS:IO>
  $<TARGET_OBJECTS:Visualization>
)
# Suppressing this since there are too many warnings
# open3d_show_and_abort_on_warning(${OPEN3D_CUDA_MODULE})
open3d_set_global_properties(${OPEN3D_CUDA_MODULE})
open3d_link_3rdparty_libraries(${OPEN3D_CUDA_MODULE})

# Build with opencv support
if(CUDA_WITH_OPENCV)
  find_package(OpenCV REQUIRED)
  message("Building CUDA with OpenCV enabled")
  target_link_libraries(${OPEN3D_CUDA_MODULE} PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(${OPEN3D_CUDA_MODULE} PUBLIC USE_OPENCV)
endif()

# target_compile_options(${OPEN3D_CUDA_MODULE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                    -g
#                    -G
#                    >)
add_library(${PROJECT_NAME}::${OPEN3D_CUDA_MODULE} ALIAS ${OPEN3D_CUDA_MODULE})

# Installation
install(
  TARGETS ${OPEN3D_CUDA_MODULE} EXPORT ${OPEN3D_CUDA_MODULE}Targets
  RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${Open3D_INSTALL_LIB_DIR}
)

install(
  EXPORT ${OPEN3D_CUDA_MODULE}Targets
  FILE "${OPEN3D_CUDA_MODULE}Targets.cmake"
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION lib/cmake/${PROJECT_NAME}
  COMPONENT ${OPEN3D_CUDA_MODULE}
)

configure_file(
  "${OPEN3D_CUDA_MODULE}Config.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${OPEN3D_CUDA_MODULE}Config.cmake"
  @ONLY
)

install(
  FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${OPEN3D_CUDA_MODULE}Config.cmake"
  DESTINATION ${Open3D_INSTALL_CMAKE_DIR}
  COMPONENT ${OPEN3D_CUDA_MODULE}
)

#install headers
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DESTINATION "${Open3D_INSTALL_INCLUDE_DIR}"
  PATTERN "*.cpp" EXCLUDE
  PATTERN "*.cuh" EXCLUDE
  PATTERN "*.cu" EXCLUDE
  PATTERN "*.txt" EXCLUDE
)
