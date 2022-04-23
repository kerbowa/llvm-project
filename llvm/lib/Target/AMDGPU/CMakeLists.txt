add_llvm_component_group(AMDGPU)

set(LLVM_TARGET_DEFINITIONS AMDGPU.td)

tablegen(LLVM AMDGPUGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AMDGPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AMDGPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AMDGPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AMDGPUGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AMDGPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AMDGPUGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AMDGPUGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM AMDGPUGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM AMDGPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AMDGPUGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM AMDGPUGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS AMDGPUGISel.td)
tablegen(LLVM AMDGPUGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM AMDGPUGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPUPreLegalizerCombinerHelper")
tablegen(LLVM AMDGPUGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPUPostLegalizerCombinerHelper")
tablegen(LLVM AMDGPUGenRegBankGICombiner.inc -gen-global-isel-combiner
              -combiners="AMDGPURegBankCombinerHelper")

set(LLVM_TARGET_DEFINITIONS R600.td)
tablegen(LLVM R600GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM R600GenCallingConv.inc -gen-callingconv)
tablegen(LLVM R600GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM R600GenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM R600GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM R600GenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM R600GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM R600GenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(AMDGPUCommonTableGen)

set(LLVM_TARGET_DEFINITIONS InstCombineTables.td)
tablegen(LLVM InstCombineTables.inc -gen-searchable-tables)
add_public_tablegen_target(InstCombineTableGen)

add_llvm_target(AMDGPUCodeGen
  AMDGPUAliasAnalysis.cpp
  AMDGPUAlwaysInlinePass.cpp
  AMDGPUAnnotateKernelFeatures.cpp
  AMDGPUAnnotateUniformValues.cpp
  AMDGPUArgumentUsageInfo.cpp
  AMDGPUAsmPrinter.cpp
  AMDGPUAtomicOptimizer.cpp
  AMDGPUCallLowering.cpp
  AMDGPUCodeGenPrepare.cpp
  AMDGPUExportClustering.cpp
  AMDGPUFixFunctionBitcasts.cpp
  AMDGPUFrameLowering.cpp
  AMDGPUHSAMetadataStreamer.cpp
  AMDGPUInstCombineIntrinsic.cpp
  AMDGPUInstrInfo.cpp
  AMDGPUInstructionSelector.cpp
  AMDGPUISelDAGToDAG.cpp
  AMDGPUISelLowering.cpp
  AMDGPUGlobalISelUtils.cpp
  AMDGPULateCodeGenPrepare.cpp
  AMDGPULegalizerInfo.cpp
  AMDGPULibCalls.cpp
  AMDGPULibFunc.cpp
  AMDGPULowerIntrinsics.cpp
  AMDGPULowerKernelArguments.cpp
  AMDGPULowerKernelAttributes.cpp
  AMDGPULowerKernelCalls.cpp
  AMDGPULowerModuleLDSPass.cpp
  AMDGPUMachineCFGStructurizer.cpp
  AMDGPUMachineFunction.cpp
  AMDGPUMachineModuleInfo.cpp
  AMDGPUMacroFusion.cpp
  AMDGPUMCInstLower.cpp
  AMDGPUMIRFormatter.cpp
  AMDGPUOpenCLEnqueuedBlockLowering.cpp
  AMDGPUPostLegalizerCombiner.cpp
  AMDGPUPreLegalizerCombiner.cpp
  AMDGPUPromoteAlloca.cpp
  AMDGPUPropagateAttributes.cpp
  AMDGPURegBankCombiner.cpp
  AMDGPURegisterBankInfo.cpp
  AMDGPURewriteOutArguments.cpp
  AMDGPUSubtarget.cpp
  AMDGPUTargetMachine.cpp
  AMDGPUTargetObjectFile.cpp
  AMDGPUTargetTransformInfo.cpp
  AMDGPUUnifyDivergentExitNodes.cpp
  AMDGPUUnifyMetadata.cpp
  AMDGPUPerfHintAnalysis.cpp
  AMDILCFGStructurizer.cpp
  AMDGPUPrintfRuntimeBinding.cpp
  AMDGPUResourceUsageAnalysis.cpp
  GCNHazardRecognizer.cpp
  GCNIterativeScheduler.cpp
  GCNMinRegStrategy.cpp
  GCNRegPressure.cpp
  GCNSchedStrategy.cpp
  R600AsmPrinter.cpp
  R600ClauseMergePass.cpp
  R600ControlFlowFinalizer.cpp
  R600EmitClauseMarkers.cpp
  R600ExpandSpecialInstrs.cpp
  R600FrameLowering.cpp
  R600InstrInfo.cpp
  R600ISelLowering.cpp
  R600MachineFunctionInfo.cpp
  R600MachineScheduler.cpp
  R600OpenCLImageTypeLoweringPass.cpp
  R600OptimizeVectorRegisters.cpp
  R600Packetizer.cpp
  R600RegisterInfo.cpp
  SIAnnotateControlFlow.cpp
  SIFixSGPRCopies.cpp
  SIFixVGPRCopies.cpp
  SIPreAllocateWWMRegs.cpp
  SIFoldOperands.cpp
  SIFormMemoryClauses.cpp
  SIFrameLowering.cpp
  SIInsertHardClauses.cpp
  SILateBranchLowering.cpp
  SIInsertWaitcnts.cpp
  SIInstrInfo.cpp
  SIISelLowering.cpp
  SILoadStoreOptimizer.cpp
  SILowerControlFlow.cpp
  SILowerI1Copies.cpp
  SILowerSGPRSpills.cpp
  SIMachineFunctionInfo.cpp
  SIMachineScheduler.cpp
  SIMemoryLegalizer.cpp
  SIOptimizeExecMasking.cpp
  SIOptimizeExecMaskingPreRA.cpp
  SIPeepholeSDWA.cpp
  SIPostRABundler.cpp
  SIPreEmitPeephole.cpp
  SIProgramInfo.cpp
  SIRegisterInfo.cpp
  SIShrinkInstructions.cpp
  SIWholeQuadMode.cpp
  GCNILPSched.cpp
  GCNRegBankReassign.cpp
  GCNNSAReassign.cpp
  GCNDPPCombine.cpp
  GCNPreRAOptimizations.cpp
  SIModeRegister.cpp
  OptSched/lib/Scheduler/aco.hip.cpp
  OptSched/lib/Scheduler/simplified_aco_ds.hip.cpp
  OptSched/lib/Scheduler/bb_spill.hip.cpp
  OptSched/lib/Scheduler/buffers.cpp
  OptSched/lib/Scheduler/config.cpp
  OptSched/lib/Scheduler/data_dep.hip.cpp
  OptSched/lib/Scheduler/enumerator.cpp
  OptSched/lib/Scheduler/gen_sched.hip.cpp
  OptSched/lib/Scheduler/graph.hip.cpp
  OptSched/lib/Scheduler/graph_trans.cpp
  OptSched/lib/Scheduler/hist_table.cpp
  OptSched/lib/Scheduler/list_sched.hip.cpp
  OptSched/lib/Scheduler/logger.cpp
  OptSched/lib/Scheduler/reg_alloc.cpp
  OptSched/lib/Scheduler/utilities.cpp
  OptSched/lib/Scheduler/machine_model.hip.cpp
  OptSched/lib/Scheduler/random.hip.cpp
  OptSched/lib/Scheduler/ready_list.hip.cpp
  OptSched/lib/Scheduler/register.hip.cpp
  OptSched/lib/Scheduler/relaxed_sched.cpp
  OptSched/lib/Scheduler/sched_basic_data.hip.cpp
  OptSched/lib/Scheduler/sched_region.hip.cpp
  OptSched/lib/Scheduler/stats.cpp
  OptSched/lib/Wrapper/OptimizingScheduler.hip.cpp
  OptSched/lib/Wrapper/OptSchedMachineWrapper.cpp
  OptSched/lib/Wrapper/OptSchedDDGWrapperBasic.cpp
  OptSched/lib/Wrapper/OptSchedGenericTarget.cpp
  OptSched/lib/Wrapper/AMDGPU/GCNOptSched.cpp
  OptSched/lib/Wrapper/AMDGPU/OptSchedGCNTarget.cpp
  OptSched/lib/Wrapper/AMDGPU/OptSchedDDGWrapperGCN.cpp


  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  IPO
  MC
  Passes
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  Scalar
  SelectionDAG
  Support
  Target
  TransformUtils
  Vectorize
  GlobalISel
  BinaryFormat
  MIRParser

  ADD_TO_COMPONENT
  AMDGPU
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
include_directories("OptSched/include" "OptSched/lib")
target_include_directories(LLVMAMDGPUCodeGen
	PRIVATE
        ${HIP_PATH}/include
        ${HIP_PATH}/../include)
message(STATUS "HIP_LIB: ${HIP_LIBRARIES}")
target_include_directories(LLVMAMDGPUCodeGen
    PRIVATE
        ${HIP_PATH}/include
        ${HIP_PATH}/../include
        ${OPTSCHED_TARGET_DEPS})


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -fgpu-rdc")
target_link_options(LLVMAMDGPUCodeGen PUBLIC -fgpu-rdc --hip-link)
target_link_libraries(LLVMAMDGPUCodeGen PRIVATE ${HIP_LIBRARIES})

set(OPTSCHED_TARGET_DEPS ${OPTSCHED_SOURCE_DIR}/include ${OPTSCHED_SOURCE_DIR}/lib)
MESSAGE(INFO ${OPTSCHED_TARGET_DEPS})

target_compile_definitions(LLVMAMDGPUCodeGen PRIVATE INSERT_ON_STEPFRWRD)
target_include_directories(LLVMAMDGPUCodeGen PRIVATE ${OPTSCHED_TARGET_DEPS})
target_link_libraries(LLVMAMDGPUCodeGen PRIVATE ${HIP_LIBRARIES})
