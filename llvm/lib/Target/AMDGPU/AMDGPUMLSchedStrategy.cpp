//===-- AMDGPUMLSchedStrategy.cpp - ML-focused Scheduler Strategy ---------===//
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

#include "AMDGPUMLSchedStrategy.h"

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

AMDGPUMLSchedStrategy::AMDGPUMLSchedStrategy(const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  // Use more accurate GCN pressure trackers.
  UseGCNTrackers = false;
  // Always schedule top-down for better blancing of HW resource usage.
  RegionPolicy.OnlyTopDown = true;
}

void AMDGPUMLSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GCNSchedStrategy::initialize(DAG);

  const MCSchedModel &SM = MF->getSubtarget().getSchedModel();
  unsigned NumPR = SM.getNumProcResourceKinds();
  HWUInfo.resize(NumPR);
  for (unsigned I = 0; I < NumPR; I++) {
    HWUInfo[I].setRes(SM.getProcResource(I));
  }
  CriticalResourceIdx = NumPR + 1;
}

static bool shouldCheckPending(SchedBoundary &Zone,
                               const TargetSchedModel *SchedModel) {
  return true;

  // FIXME -- enable this method, need to share flag
  // bool HasBufferedModel =
  //    SchedModel->hasInstrSchedModel() && SchedModel->getMicroOpBufferSize();
  // unsigned Combined = Zone.Available.size() + Zone.Pending.size();
  // return true; //Combined <= PendingQueueLimit && HasBufferedModel;
}

static SUnit *pickOnlyChoice(SchedBoundary &Zone,
                             const TargetSchedModel *SchedModel) {
  // pickOnlyChoice() releases pending instructions and checks for new hazards.
  SUnit *OnlyChoice = Zone.pickOnlyChoice();
  if (!shouldCheckPending(Zone, SchedModel) || Zone.Pending.empty())
    return OnlyChoice;

  return nullptr;
}

void AMDGPUMLSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  auto MI = SU->getInstr();
  const SIInstrInfo *SII = reinterpret_cast<const SIInstrInfo *>(DAG->TII);

  if (SchedModel && SchedModel->hasInstrSchedModel()) {
    const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
    for (TargetSchedModel::ProcResIter
             PI = SchedModel->getWriteProcResBegin(SC),
             PE = SchedModel->getWriteProcResEnd(SC);
         PI != PE; ++PI) {
      HWUInfo[PI->ProcResourceIdx].schedule(SU, PI->ReleaseAtCycle);
    }

    updateCriticalResource();

    if (SII->isMFMAorWMMA(*MI)) {
      SchedMFMA.push_back(SU);
    }
    if (SII->isDS(*MI) && MI->mayLoad()) {
      SchedDSR.push_back(SU);
    }
  }

  GCNSchedStrategy::schedNode(SU, IsTopNode);
}

void AMDGPUMLSchedStrategy::updateCriticalResource() {
  unsigned MaxCycles = 0;

  unsigned I = 0;
  bool Updated = false;

  for (auto &HWUI : HWUInfo) {
    if (I == 4) {
      I++;
      continue;
    }
    if (HWUI.getTotalCycles() > MaxCycles) {
      assert(HWUI.getProcRes() && "Missing resource?");
      CriticalResourceIdx = I;
      MaxCycles = HWUI.getTotalCycles();
    }
    I++;
  }
  unsigned SecondaryCycles = 0;
  I = 0;

  for (auto &HWUI : HWUInfo) {
    if (I == 4) {
      I++;
      continue;
    }
    if (HWUI.getTotalCycles() > SecondaryCycles &&
        HWUI.getTotalCycles() <= MaxCycles && CriticalResourceIdx != I) {
      assert(HWUI.getProcRes() && "Missing resource?");
      SecondaryResourceIdx = I;
      Updated = true;
      SecondaryCycles = HWUI.getTotalCycles();
    }
    I++;
  }

  if (!Updated) {
    SecondaryResourceIdx = CriticalResourceIdx;
  }
}

void AMDGPUMLSchedStrategy::collectUse() {
  CollectedUse = true;
  SchedDSR.clear();
  SchedMFMA.clear();

  if (!SchedModel || !SchedModel->hasInstrSchedModel())
    return;

  for (auto &SU : DAG->SUnits) {
    const MCSchedClassDesc *SC = DAG->getSchedClass(&SU);
    for (TargetSchedModel::ProcResIter
             PI = SchedModel->getWriteProcResBegin(SC),
             PE = SchedModel->getWriteProcResEnd(SC);
         PI != PE; ++PI) {
      auto Opc = SU.getInstr()->getOpcode();
      bool IsDMA = Opc == AMDGPU::TENSOR_LOAD_TO_LDS_D2 ||
                   Opc == AMDGPU::TENSOR_LOAD_TO_LDS_D2_gfx1250 ||
                   Opc == AMDGPU::TENSOR_LOAD_TO_LDS ||
                   Opc == AMDGPU::TENSOR_LOAD_TO_LDS_gfx1250 ||
                   Opc == AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B32 ||
                   Opc == AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B32_gfx1250 ||
                   Opc == AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B32_SADDR ||
                   Opc == AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B32_SADDR_gfx1250;
      unsigned Latency = IsDMA ? SU.Latency : PI->ReleaseAtCycle;
      HWUInfo[PI->ProcResourceIdx].insert(&SU, Latency);
    }
  }

  updateCriticalResource();
}

bool AMDGPUMLSchedStrategy::tryCriticalResource(SchedCandidate &TryCand,
                                                SchedCandidate &Cand,
                                                SchedBoundary *Zone) const {
  if (CriticalResourceIdx == SchedModel->getNumProcResourceKinds() + 1)
    return false;

  unsigned MaxAvailableLat = Zone->findMaxLatency(Zone->Available.elements());

  HardwareUnitInfo HWUI = HWUInfo[CriticalResourceIdx];
  unsigned CriticalUsage = HWUI.getTotalCycles();

  if (MaxAvailableLat > CriticalUsage)
    return false;

  bool CandUsesCrit = HWUI.contains(Cand.SU);
  bool TryCandUsesCrit = HWUI.contains(TryCand.SU);

  if (!CandUsesCrit && !TryCandUsesCrit)
    return false;

  if (CandUsesCrit && !TryCandUsesCrit) {
    if (Cand.Reason > RegCritical)
      Cand.Reason = RegCritical;
    return true;
  }

  if (!CandUsesCrit && TryCandUsesCrit) {
    TryCand.Reason = RegCritical;
    return true;
  }

  if (SecondaryResourceIdx != CriticalResourceIdx &&
      tryCriticalResourceDependency(TryCand, Cand, Zone,
                                    SecondaryResourceIdx)) {
    return true;
  }

  if (HWUI.isHigherPriority(Cand.SU, TryCand.SU)) {
    if (Cand.Reason > RegCritical)
      Cand.Reason = RegCritical;
    return true;
  }

  TryCand.Reason = RegCritical;
  return true;
}

bool AMDGPUMLSchedStrategy::tryCriticalResourceDependency(
    SchedCandidate &TryCand, SchedCandidate &Cand, SchedBoundary *Zone,
    unsigned ResourceIdx) const {
  if (ResourceIdx == SchedModel->getNumProcResourceKinds() + 1)
    return false;

  unsigned MaxAvailableLat = Zone->findMaxLatency(Zone->Available.elements());
  HardwareUnitInfo HWUI = HWUInfo[ResourceIdx];
  unsigned CriticalUsage = HWUI.getTotalCycles();

  if (MaxAvailableLat > CriticalUsage)
    return false;

  auto *TargetSU = HWUI.getNextTargetSU();
  if (!TargetSU)
    return false;

  bool CandEnables = DAG->IsReachable(TargetSU, Cand.SU);
  bool TryCandEnables = DAG->IsReachable(TargetSU, TryCand.SU);

  if (!CandEnables && !TryCandEnables)
    return false;

  if (CandEnables && !TryCandEnables) {
    if (Cand.Reason > RegCritical)
      Cand.Reason = RegCritical;

    return true;
  }

  if (!CandEnables && TryCandEnables) {
    TryCand.Reason = RegCritical;
    return true;
  }

  // Both enable, prefer the critical path.
  bool CandHeight = Cand.SU->getHeight();
  bool TryCandHeight = TryCand.SU->getHeight();

  if (CandHeight > TryCandHeight) {
    if (Cand.Reason > RegCritical)
      Cand.Reason = RegCritical;

    return true;
  }

  if (CandHeight < TryCandHeight) {
    TryCand.Reason = RegCritical;
    return true;
  }

  // Same critical path, just prefer original candidate.
  if (Cand.Reason > RegCritical)
    Cand.Reason = RegCritical;

  return true;
}

unsigned
AMDGPUMLSchedStrategy::getLatencyStallCycles(SUnit *SU,
                                             unsigned CurrCycle) const {
  unsigned ReadyCycle = SU->TopReadyCycle;
  auto *MI = SU->getInstr();
  const SIInstrInfo *SII = reinterpret_cast<const SIInstrInfo *>(DAG->TII);

  if (SII->isDS(*MI) && MI->mayLoad()) {
    if (SchedDSR.size() >= 8) {
      unsigned TopOfFIFO = SchedDSR.size() - 8;
      unsigned TopOfFIFOIssue = SchedDSR[TopOfFIFO]->TopReadyCycle;
      // TODO -- should be release at cycle.
      ReadyCycle = std::max(TopOfFIFOIssue + 20, ReadyCycle);
    }
  }

  else if (SII->isMFMAorWMMA(*MI) && SchedMFMA.size()) {
    auto PrevMFMA = SchedMFMA[SchedMFMA.size() - 1];
    unsigned PrevMFMAIssue = PrevMFMA->TopReadyCycle;
    ReadyCycle = std::max(PrevMFMAIssue + PrevMFMA->Latency, ReadyCycle);
  }

  if (ReadyCycle > CurrCycle)
    return ReadyCycle - CurrCycle;
  return 0;
}

bool AMDGPUMLSchedStrategy::tryPendingCandidate(SchedCandidate &Cand,
                                                SchedCandidate &TryCand,
                                                SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  // Avoid exceeding the target's limit.
  /*if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;*/

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(getLatencyStallCycles(TryCand.SU, Zone->getCurrCycle()),
                getLatencyStallCycles(Cand.SU, Zone->getCurrCycle()), TryCand,
                Cand, Stall))
      return TryCand.Reason != NoCand;

    if (tryCriticalResource(TryCand, Cand, Zone)) {
      return TryCand.Reason != NoCand;
    }
  }

  return false;
}

bool AMDGPUMLSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                         SchedCandidate &TryCand,
                                         SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  // Avoid exceeding the target's limit.
  /*
  if (DAG->isTrackingPressure() && tryPressure(TryCand.RPDelta.Excess,
                                               Cand.RPDelta.Excess,
                                               TryCand, Cand, RegExcess, TRI,
                                               DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() && tryPressure(TryCand.RPDelta.CriticalMax,
                                               Cand.RPDelta.CriticalMax,
                                               TryCand, Cand, RegCritical, TRI,
                                               DAG->MF))
    return TryCand.Reason != NoCand;
*/
  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary. Some properties are simply incomparable, in many
  // other instances we should only override the other boundary if something
  // is a clear good pick on one boundary. Skip heuristics that are more
  // "tie-breaking" in nature.
  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {

    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(getLatencyStallCycles(TryCand.SU, Zone->getCurrCycle()),
                getLatencyStallCycles(Cand.SU, Zone->getCurrCycle()), TryCand,
                Cand, Stall))
      return TryCand.Reason != NoCand;

    if (tryCriticalResource(TryCand, Cand, Zone)) {
      return TryCand.Reason != NoCand;
    }

    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    if (tryCriticalResourceDependency(TryCand, Cand, Zone,
                                      CriticalResourceIdx)) {
      return TryCand.Reason != NoCand;
    }
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  unsigned CandZoneCluster = getClusterID(Cand.AtTop);
  unsigned TryCandZoneCluster = getClusterID(TryCand.AtTop);
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce)) {
      return TryCand.Reason != NoCand;
    }
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand)) {
      return TryCand.Reason != NoCand;
    }

    // Avoid serializing long latency dependence chains.
    // For acyclic path limited loops, latency was already checked above.
    if (!RegionPolicy.DisableLatencyHeuristic && TryCand.Policy.ReduceLatency &&
        !Rem.IsAcyclicLatencyLimited && tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

void AMDGPUMLSchedStrategy::pickNodeFromQueue(
    SchedBoundary &Zone, const CandPolicy &ZonePolicy,
    const RegPressureTracker &RPTracker, SchedCandidate &Cand, bool &IsPending,
    bool IsBottomUp) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  IsPending = false;
  if (DAG->isTrackingPressure()) {
    if (!UseGCNTrackers) {
      SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
      VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
    } else {
      GCNRPTracker *T = IsBottomUp
                            ? static_cast<GCNRPTracker *>(&UpwardTracker)
                            : static_cast<GCNRPTracker *>(&DownwardTracker);
      SGPRPressure = T->getPressure().getSGPRNum();
      VGPRPressure = T->getPressure().getArchVGPRNum();
    }
  }
  LLVM_DEBUG(dbgs() << "Available Q:\n");
  ReadyQueue &AQ = Zone.Available;
  for (SUnit *SU : AQ) {
    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    tryCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      LLVM_DEBUG(printCandidateDecision(Cand, TryCand));
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }

  if (!shouldCheckPending(Zone, SchedModel))
    return;

  LLVM_DEBUG(dbgs() << "Pending Q:\n");
  ReadyQueue &PQ = Zone.Pending;
  for (SUnit *SU : PQ) {
    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    AMDGPUMLSchedStrategy::tryPendingCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      LLVM_DEBUG(printCandidateDecision(Cand, TryCand));
      IsPending = true;
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }
}

SUnit *AMDGPUMLSchedStrategy::pickNode(bool &IsTopNode) {
  if (!CollectedUse)
    collectUse();

  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  bool PickedPending;
  SUnit *SU;
  do {
    PickedPending = false;
    if (RegionPolicy.OnlyTopDown) {
      SU = pickOnlyChoice(Top, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, TopCand.Policy, DAG->getTopRPTracker(), TopCand,
                          PickedPending,
                          /*IsBottomUp=*/false);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = pickOnlyChoice(Bot, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, BotCand.Policy, DAG->getBotRPTracker(), BotCand,
                          PickedPending,
                          /*IsBottomUp=*/true);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode, PickedPending);
    }
  } while (SU->isScheduled);

  if (PickedPending) {
    unsigned ReadyCycle = IsTopNode ? SU->TopReadyCycle : SU->BotReadyCycle;
    SchedBoundary &Zone = IsTopNode ? Top : Bot;
    unsigned CurrentCycle = Zone.getCurrCycle();
    if (ReadyCycle > CurrentCycle)
      Zone.bumpCycle(ReadyCycle);

    // FIXME: checkHazard() doesn't give information about which cycle the
    // hazard will resolve so just keep bumping the cycle by 1. This could be
    // made more efficient if checkHazard() returned more details.
    while (Zone.checkHazard(SU))
      Zone.bumpCycle(Zone.getCurrCycle() + 1);

    Zone.releasePending();
  }

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  return SU;
}

AMDGPUMLPostSchedStrategy::AMDGPUMLPostSchedStrategy(
    const MachineSchedContext *C)
    : PostGenericScheduler(C) {}

bool AMDGPUMLPostSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                             SchedCandidate &TryCand) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Prioritize instructions that read unbuffered resources by stall cycles.
  if (tryLess(Top.getLatencyStallCycles(TryCand.SU),
              Top.getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
    return TryCand.Reason != NoCand;

  // Keep clustered nodes together.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;
  // Avoid critical resource consumption and balance the schedule.
  if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
              TryCand, Cand, ResourceReduce))
    return TryCand.Reason != NoCand;
  if (tryGreater(TryCand.ResDelta.DemandedResources,
                 Cand.ResDelta.DemandedResources, TryCand, Cand,
                 ResourceDemand))
    return TryCand.Reason != NoCand;

  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary.
  if (Cand.AtTop == TryCand.AtTop) {
    // Avoid serializing long latency dependence chains.
    if (Cand.Policy.ReduceLatency &&
        tryLatency(TryCand, Cand, Cand.AtTop ? Top : Bot))
      return TryCand.Reason != NoCand;
  }

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum < Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  return false;
}
