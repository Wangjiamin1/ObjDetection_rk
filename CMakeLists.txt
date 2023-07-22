cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# install target and libraries
#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# set(MZ_ROOT "/home/wjm/rknn_model_zoo")
# set(TARGET_SOC "rk3588")

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# rknn api
set(PLATFORM RK3588)
set(LIB_ARCH aarch64)
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/libs/rklibs/rknpu2/runtime/RK3588/Linux/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
set(RKNN_DRM_LIB ${CMAKE_SOURCE_DIR}/libs/rklibs/librga/samples/utils/3rdparty/libdrm/lib/arm64/libdrm.so)
set(RGA_PATH ${CMAKE_SOURCE_DIR}/libs/rklibs/librga)


include_directories(${RKNN_API_PATH}/include)


# drm
set(DRM_DIR ${CMAKE_SOURCE_DIR}/libs/common/drm)
include_directories(${DRM_DIR}/include)
include_directories(${DRM_DIR}/include/libdrm)


#stb CImg
set(STB_DIR ${CMAKE_SOURCE_DIR}/libs/common/)
include_directories(${STB_DIR})


# libpng
set(LIBPNG_DIR ${CMAKE_SOURCE_DIR}/libs/libpng)
include_directories(${LIBPNG_DIR}/include)
set(LIBPNG_LIB ${LIBPNG_DIR}/lib/Linux/libpng16.a)
# libjpeg
set(LIBJPEG_DIR ${CMAKE_SOURCE_DIR}/libs/libjpeg)
include_directories(${LIBJPEG_DIR}/include)
set(LIBJPEG_LIB ${LIBJPEG_DIR}/lib/Linux/libjpeg.a)
# zlib(used by libpng)
set(ZLIB_DIR ${CMAKE_SOURCE_DIR}/libs/zlib)
include_directories(${ZLIB_DIR}/include)
set(ZLIB_LIB ${ZLIB_DIR}/lib/Linux/libz.a)




set(RGA_LIB ${RGA_PATH}/libs/Linux/gcc-${LIB_ARCH}/librga.so)
include_directories(${RGA_PATH}/include)

string(REPLACE "-DANDROID" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-DANDROID" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

set(CMAKE_INSTALL_RPATH "lib")


# ObjDetction
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(ObjDetction
        src/drm_func.c
        src/rga_func.c
        src/yolo.cc
        src/main.cc
        src/tracker.cpp
)

target_link_libraries(ObjDetction -lyaml-cpp
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${LIBJPEG_LIB}
  ${LIBPNG_LIB}
  ${ZLIB_LIB}
  ${RKNN_DRM_LIB}
  dl
  ${OpenCV_LIBS}
)

# install target and libraries
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/)
# install(TARGETS ObjDetction DESTINATION ./)

# install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
# install(PROGRAMS ${RKNN_DRM_LIB} DESTINATION lib)
# install(PROGRAMS ${RGA_LIB} DESTINATION lib)
# install(PROGRAMS /lib/libx264.so.164 DESTINATION lib)
# install(DIRECTORY model DESTINATION ./)



