//===- AMDGPUPreloadKernargHeader.cpp - Header for preloaded kernargs  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// TODO: Rewrite
/// When doing kernarg preloading we must ensure backward compatibility with
/// older firmware versions by adding a section of code at the kernel entry to
/// facilitate this compatibility layer. Firmware that does support kernarg
/// preloading expects 256-bytes of padding at the kernel entry. Some more
/// recent hardware does not need this compatibility layer. However, Padding
/// should be generated regardless of the need for backward compatibility.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-preload-kernarg-header"

namespace {

class AMDGPUPreloadKernargHeader : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreloadKernargHeader() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
    if (!Info->isEntryFunction() || !Info->getNumKernargPreloadedSGPRs())
      return false;

    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    if (ST.needsKernargPreloadBackwardsCompatibility()) {
      MachineBasicBlock *BCH = Info->BCH;
      MachineBasicBlock *PH = Info->PH;
      assert(BCH);
      SmallVector<MachineInstr *, 4> Worklist;
      for (auto &MI : *BCH) {
        if (MI.getOpcode() == TargetOpcode::INLINEASM_BR)
          Worklist.push_back(&MI);
      }

      assert(PH);
      for (auto &MI : *PH) {
        if (MI.getOpcode() == TargetOpcode::INLINEASM_BR)
          Worklist.push_back(&MI);
      }

      for (auto *MI : Worklist) {
        MI->eraseFromParent();
      }

      BuildMI(*BCH, BCH->end(), DebugLoc(), TII->get(AMDGPU::S_BRANCH))
          .addMBB(Info->KernStart);

      unsigned NumBytes = 0;
      for (auto &MI : *BCH) {
        NumBytes += TII->getInstSizeInBytes(MI);
      }

      unsigned RequiredPadding = 256 - NumBytes;
      while (RequiredPadding >= 4) {
        BuildMI(*BCH, BCH->end(), DebugLoc(), TII->get(AMDGPU::S_NOP))
            .addImm(0);
        RequiredPadding -= 4;
      }
      assert(RequiredPadding == 0);
    } else {
      MachineBasicBlock &FirstMBB = MF.front();
      for (int I = 0; I < 64; ++I) {
        BuildMI(FirstMBB, FirstMBB.begin(), DebugLoc(), TII->get(AMDGPU::S_NOP))
            .addImm(0);
      }
    }

    return true;
  }
};

} // namespace

char AMDGPUPreloadKernargHeader::ID = 0;

char &llvm::AMDGPUPreloadKernargHeaderID = AMDGPUPreloadKernargHeader::ID;

INITIALIZE_PASS(AMDGPUPreloadKernargHeader, DEBUG_TYPE, "AMDGPU Preload Kernarg Header",
                false, false)
