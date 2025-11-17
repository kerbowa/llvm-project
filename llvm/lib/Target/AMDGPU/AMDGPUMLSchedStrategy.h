//===-- AMDGPUMLSchedStrategy.h - ML-focused Scheduler Strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// ML-focused scheduling strategy for AMDGPU.
//
//===----------------------------------------------------------------------===//

#include "GCNSchedStrategy.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class HardwareUnitInfo {
private:
  const MCProcResourceDesc *ProcRes = nullptr;
  SmallPtrSet<SUnit *, 16> PrioritySUs;
  SmallPtrSet<SUnit *, 16> AllSUs;
  unsigned TotalCycles = 0;

public:
  HardwareUnitInfo(const MCProcResourceDesc *Res) : ProcRes(Res) {};
  HardwareUnitInfo() {}

  void setRes(const MCProcResourceDesc *Res) { ProcRes = Res; }

  unsigned size() { return AllSUs.size(); }
  SUnit *getTargetSU() { return *PrioritySUs.begin(); }
  SUnit *getNextTargetSU() {
    for (auto *PrioritySU : PrioritySUs) {
      if (!PrioritySU->isTopReady())
        return PrioritySU;
    }
    return nullptr;
  }

  unsigned getTotalCycles() { return TotalCycles; }
  const MCProcResourceDesc *getProcRes() { return ProcRes; }

  void insert(SUnit *SU, unsigned ReleaseAtCycle) {
    auto Inserted = AllSUs.insert(SU);
    TotalCycles += ReleaseAtCycle;

    // errs() << "TotalCycles increased to: " << TotalCycles << "\n";

    assert(Inserted.second);
    if (PrioritySUs.empty()) {
      PrioritySUs.insert(SU);
      return;
    }
    unsigned SUDepth = SU->getDepth();
    unsigned CurrDepth = (*PrioritySUs.begin())->getDepth();
    if (SUDepth > CurrDepth)
      return;

    if (SUDepth == CurrDepth) {
      PrioritySUs.insert(SU);
      return;
    }

    // SU is lower depth and should be prioritized.
    PrioritySUs.clear();
    PrioritySUs.insert(SU);
  }

  bool contains(SUnit *SU) { return AllSUs.contains(SU); }

  bool isHigherPriority(SUnit *SU, SUnit *Other) {
    for (auto *SUOrder : PrioritySUs) {
      if (SUOrder == SU)
        return true;
      if (SUOrder == Other)
        return false;
    }

    return false;
  }

  void schedule(SUnit *SU, unsigned ReleaseAtCycle) {
    AllSUs.erase(SU);
    PrioritySUs.erase(SU);
    TotalCycles -= ReleaseAtCycle;
    if (AllSUs.empty())
      return;
    if (PrioritySUs.empty()) {
      for (auto SU : AllSUs) {
        if (PrioritySUs.empty()) {
          PrioritySUs.insert(SU);
          continue;
        }
        unsigned SUDepth = SU->getDepth();
        unsigned CurrDepth = (*PrioritySUs.begin())->getDepth();
        if (SUDepth > CurrDepth)
          continue;

        if (SUDepth == CurrDepth) {
          PrioritySUs.insert(SU);
          continue;
        }

        // SU is lower depth and should be prioritized.
        PrioritySUs.clear();
        PrioritySUs.insert(SU);
      }
    }
  }

  void reset() {
    AllSUs.clear();
    PrioritySUs.clear();
    TotalCycles = 0;
  }
};

class AMDGPUMLSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

  SmallVector<SUnit *, 16> SchedDSR;

  SmallVector<SUnit *, 16> SchedMFMA;

  SmallVector<HardwareUnitInfo, 8> HWUInfo;

  void collectUse();

  void updateCriticalResource();

  bool tryPendingCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                           SchedBoundary *Zone) const;

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand, bool &IsPending,
                         bool IsBottomUp);

  SUnit *pickNode(bool &IsTopNode) override;

public:
  AMDGPUMLSchedStrategy(const MachineSchedContext *C);

  void initialize(ScheduleDAGMI *DAG) override;

  void schedNode(SUnit *SU, bool IsTopNode) override;

  bool tryCriticalResource(SchedCandidate &TryCand, SchedCandidate &Cand,
                           SchedBoundary *Zone) const;

  bool tryCriticalResourceDependency(SchedCandidate &TryCand,
                                     SchedCandidate &Cand, SchedBoundary *Zone,
                                     unsigned ResourceIdx) const;

  unsigned getLatencyStallCycles(SUnit *SU, unsigned CurrCycle) const;

  unsigned CriticalResourceIdx;
  unsigned SecondaryResourceIdx;
};

class AMDGPUMLPostSchedStrategy : public PostGenericScheduler {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand) override;

public:
  AMDGPUMLPostSchedStrategy(const MachineSchedContext *C);
};

} // End namespace llvm