macro(add_filskalang_subdirectory name)
  add_llvm_subdirectory(FILSKALANG TOOL ${name})
endmacro()

macro(add_filskalang_library name)
  if(BUILD_SHARED_LIBS)
    set(LIBTYPE SHARED)
  else()
    set(LIBTYPE STATIC)
  endif()
  llvm_add_library(${name} ${LIBTYPE} ${ARGN})
  if(TARGET ${name})
    get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

    set(LIBS
      ${dialect_libs}
      ${conversion_libs}
      MLIRAnalysis
      MLIRCallInterfaces
      MLIRCastInterfaces
      MLIRExecutionEngine
      MLIRIR
      MLIRLLVMCommonConversion
      MLIRLLVMToLLVMIRTranslation
      MLIRMemRefDialect
      MLIRLLVMDialect
      MLIRParser
      MLIRPass
      MLIRSideEffectInterfaces
      MLIRSupport
      MLIRTargetLLVMIRExport
      MLIRTransforms
      MLIROptLib
      )

    target_link_libraries(${name} INTERFACE
      ${LLVM_COMMON_LIBS}
      ${LIBS})

    install(TARGETS ${name}
      COMPONENT ${name}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      RUNTIME DESTINATION bin)
  else()
    add_custom_target(${name})
  endif()
endmacro()

macro(add_filskalang_executable name)
  add_llvm_executable(${name} ${ARGN} )
endmacro()

macro(add_filskalang_tool name)
  add_filskalang_executable(${name} ${ARGN})
  install(TARGETS ${name}
    RUNTIME DESTINATION bin
    COMPONENT ${name})
endmacro()
