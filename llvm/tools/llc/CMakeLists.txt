set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmParser
  AsmPrinter
  CodeGen
  Core
  IRReader
  MC
  MIRParser
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TransformUtils
  Vectorize
  )

add_llvm_tool(llc
  llc.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )

####################################################################
# OptSched
####################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -fgpu-rdc")
target_link_options(llc PUBLIC -fgpu-rdc --hip-link)
target_link_libraries(llc PRIVATE ${HIP_LIBRARIES})
export_executable_symbols_for_plugins(llc)

####################################################################