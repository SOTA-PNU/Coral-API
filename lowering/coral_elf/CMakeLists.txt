# Coral NPU RV32 ELF — EmitC 경로.
#
# IREE 체크아웃을 수정하지 않는다. 서브프로젝트로만 참조한다.
#
#   cmake -S coral_elf -B build/<model>/elf -G Ninja \
#     -DCMAKE_TOOLCHAIN_FILE=<iree>/build_tools/cmake/generic_riscv32.cmake \
#     -DRISCV_TOOLCHAIN_ROOT=<coral bazel toolchain> \
#     -DMODEL_DIR=<abs>/build/<model> \
#     -DCORAL_CRT=<libcrt.lo> -DCORAL_LINKER_SCRIPT=<generated.ld>
cmake_minimum_required(VERSION 3.21)
project(coral_model_elf C CXX ASM)

set(IREE_REPO_ROOT "/workspace/iree" CACHE PATH "IREE 소스 (읽기 전용)")
set(MODEL_DIR "" CACHE PATH "export_tosa.py + iree-compile 산출물 디렉터리")
set(TARGET_NAME "model_coral_elf" CACHE STRING "")

if(NOT MODEL_DIR)
  message(FATAL_ERROR "-DMODEL_DIR=<dir> 필요 (model.o, model.h, model_emitc.h)")
endif()
# VM 종류 선택.
#   model.vmfb 가 있으면  -> 바이트코드 VM (상류 static_library 샘플의 기본 경로)
#   없으면                -> EmitC (vm-c)
#
# 바이트코드를 기본으로 삼는 이유: EmitC 가 생성하는 호출 shim 은 VM ABI 인자
# 버퍼의 iree_vm_ref_t 슬롯을 항상 8바이트 정렬로 굽는데, 런타임의
# iree_vm_abi_*_t 는 평범한 C 구조체라 32비트에서 정렬이 4다. 그래서 RV32 에서
# ref 를 4바이트 어긋난 위치에서 읽어 "ref type mismatch" 가 난다.
# 바이트코드 VM 은 런타임이 자기 구조체로 직접 마샬링하므로 이 문제가 없다.
if(EXISTS "${MODEL_DIR}/model.vmfb")
  set(MODEL_VM_KIND "bytecode")
  set(_required model.o model.h manifest.json model.vmfb)
else()
  set(MODEL_VM_KIND "emitc")
  set(_required model.o model.h model_emitc.h manifest.json)
endif()
message(STATUS "MODEL_VM_KIND = ${MODEL_VM_KIND}")
foreach(_f ${_required})
  if(NOT EXISTS "${MODEL_DIR}/${_f}")
    message(FATAL_ERROR "${MODEL_DIR}/${_f} 가 없습니다")
  endif()
endforeach()
# 커널 정적 라이브러리. 바이트코드 판은 model_bc.o, EmitC 판은 model.o.
set(MODEL_KERNEL_OBJ "${MODEL_DIR}/model.o")

# ---------------------------------------------------------------------------
# IREE 런타임만. 컴파일러/샘플/테스트 없이.
# ---------------------------------------------------------------------------
set(CMAKE_DL_LIBS "")
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_RUNTIME_TRACING OFF CACHE BOOL "" FORCE)
# 베어메탈: pthread 가 없다. 이건 C 매크로가 아니라 CMake 옵션이며
# IREE 가 자기 소스 전체에 -DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 을 붙여준다.
set(IREE_SYNCHRONIZATION_DISABLE_UNSAFE ON CACHE BOOL "" FORCE)
set(IREE_FILE_IO_ENABLE OFF CACHE BOOL "" FORCE)
set(IREE_TARGET_BACKEND_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)

# 이 IREE 리비전은 알 수 없는 시스템(Generic)에 대해 정의되지 않은
# iree::async::platform::posix 를 참조한다. inline/static 경로는 이 백엔드를
# 링크도 호출도 하지 않으므로 타깃만 있는 placeholder 로 configure 를 통과시킨다.
# (/workspace/iree 는 무수정 유지)
if(CMAKE_SYSTEM_NAME STREQUAL "Generic" AND
   NOT TARGET iree::async::platform::posix)
  add_library(_iree_async_posix_placeholder INTERFACE IMPORTED GLOBAL)
  add_library(iree::async::platform::posix ALIAS _iree_async_posix_placeholder)
endif()

# generic_riscv32.cmake 이 -DIREE_PLATFORM_GENERIC / FILE_IO / TIME_NOW /
# DEVICE_SIZE_T 를 이미 넣어준다. 빠진 것만 여기서 보탠다.
#   IREE_SOCKETS_ENABLE : 순수 C 매크로. GENERIC 에서 자동 0 이 되지 않는다.
#                         (config.h 는 WASM 만 0 으로 처리)
# IREE 서브프로젝트는 자기 옵션에서 이 매크로들을 스스로 붙이지만,
# 우리 타깃(launcher.c 등)은 IREE 헤더를 직접 include 하므로 같은 정의를
# 받아야 한다. 불일치하면 pthread_mutex_t 같은 타입에서 깨진다.
if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
  add_compile_definitions(
    IREE_SOCKETS_ENABLE=0
    IREE_SYNCHRONIZATION_DISABLE_UNSAFE=1
  )
endif()

add_subdirectory("${IREE_REPO_ROOT}" iree EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# EmitC 모듈: IREE 의 module_impl_emitc.c 가 우리 헤더를 include 한다.
# 우리가 직접 include 하는 게 아니다. (build_tools/cmake/iree_c_module.cmake 방식)
# ---------------------------------------------------------------------------
if(MODEL_VM_KIND STREQUAL "emitc")
add_library(model_emitc_module STATIC
  "${IREE_REPO_ROOT}/runtime/src/iree/vm/module_impl_emitc.c"
)
target_include_directories(model_emitc_module PUBLIC "${MODEL_DIR}")
target_compile_definitions(model_emitc_module PRIVATE
  "EMITC_IMPLEMENTATION=\"model_emitc.h\""
)
target_compile_options(model_emitc_module PRIVATE
  -Wno-unused-but-set-variable      # 생성 코드에서 나오는 경고
)
target_link_libraries(model_emitc_module PUBLIC iree::vm)
else()
  # 바이트코드 판: VMFB 를 objcopy 로 .o 로 만들어 DDR(.ddr_data) 에 놓는다.
  # iree-c-embed-data 를 쓰지 않으므로 100 MB 제한과도 무관하다.
  set(VMFB_OBJ "${CMAKE_CURRENT_BINARY_DIR}/model_vmfb.o")
  find_program(OBJCOPY_BIN NAMES riscv32-unknown-elf-objcopy llvm-objcopy objcopy
               HINTS "${RISCV_TOOLCHAIN_ROOT}/bin" REQUIRED)
  add_custom_command(
    OUTPUT "${VMFB_OBJ}"
    COMMAND "${OBJCOPY_BIN}" -I binary -O elf32-littleriscv -B riscv
            --rename-section .data=.ddr_data,contents,alloc,load,readonly,data
            model.vmfb "${VMFB_OBJ}"
    WORKING_DIRECTORY "${MODEL_DIR}"
    DEPENDS "${MODEL_DIR}/model.vmfb"
    VERBATIM)
  add_custom_target(model_vmfb_obj DEPENDS "${VMFB_OBJ}")
endif()

# ---------------------------------------------------------------------------
# launcher 가 쓰는 모델별 상수. gen_shapes.py 가 manifest 로부터 만든다.
# ---------------------------------------------------------------------------
set(SHAPES_H "${CMAKE_CURRENT_BINARY_DIR}/model_shapes.h")
add_custom_command(
  OUTPUT "${SHAPES_H}"
  COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/gen_shapes.py"
          --build-dir "${MODEL_DIR}" --output "${SHAPES_H}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gen_shapes.py"
          "${MODEL_DIR}/manifest.json" "${MODEL_DIR}/model.h"
  VERBATIM
)

# model.o (RV32 커널) 는 CMake 가 그대로 링커에 넘긴다.
add_executable(${TARGET_NAME}
  launcher.c
  bare_metal_stubs.c          # IREE 의 GENERIC-as-POSIX 구멍 메우기
  "${SHAPES_H}"
  "${MODEL_KERNEL_OBJ}"
)
target_include_directories(${TARGET_NAME} PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}" "${MODEL_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE
  iree::runtime
  iree::hal::drivers::local_sync::sync_driver
  iree::hal::local::loaders::static_library_loader
)
if(MODEL_VM_KIND STREQUAL "emitc")
  target_link_libraries(${TARGET_NAME} PRIVATE model_emitc_module)
else()
  target_compile_definitions(${TARGET_NAME} PRIVATE MODEL_USE_BYTECODE=1)
  target_sources(${TARGET_NAME} PRIVATE "${VMFB_OBJ}")
  add_dependencies(${TARGET_NAME} model_vmfb_obj)
  target_link_libraries(${TARGET_NAME} PRIVATE iree::vm::bytecode::module)
endif()

# ---------------------------------------------------------------------------
# Coral CRT + 링커 스크립트 (있을 때만)
# ---------------------------------------------------------------------------
if(DEFINED CORAL_CRT AND DEFINED CORAL_LINKER_SCRIPT)
  target_link_libraries(${TARGET_NAME} PRIVATE
    "-Wl,-whole-archive" "${CORAL_CRT}" "-Wl,-no-whole-archive")
  target_link_options(${TARGET_NAME} PRIVATE
    -nostartfiles
    "-Wl,-T,${CORAL_LINKER_SCRIPT}"
    "-Wl,--gc-sections"
    # aligned_alloc 을 malloc 위에 직접 구현했으므로(bare_metal_stubs.c)
    # free 도 함께 가로채야 짝이 맞는다. newlib 의 free 는 우리가 돌려준
    # 오프셋 포인터를 해제할 수 없다.
    "-Wl,--wrap=free"
    "-Wl,--wrap=malloc"
    "-Wl,--wrap=calloc"
    "-Wl,--wrap=realloc"
    "-Wl,-Map,${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.map")
  set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
    LINK_DEPENDS "${CORAL_LINKER_SCRIPT}")
endif()
