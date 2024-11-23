//===-- GCNBalancingScheduler.cpp - Objective Balancing Scheduler ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a prototype implementation of a more balanced scheduling
/// approach for the AMDGPU backend. Instead of locking into maximum occupancy
/// at the outset, multiple candidate schedules are generated using different
/// heuristic priority configurations. A cost function then selects the best
/// candidate based on a weighted trade-off between occupancy, schedule length,
/// and (in future) additional factors like memory performance or clustering.
///
/// The goal is to:
///   - Reduce over-reliance on occupancy-first scheduling.
///   - Make the scheduler more tunable and debuggable.
///   - Provide scaffolding for workload-specific and pluggable strategies.
///   - Allow future integration with profile-guided or learned strategy selection.
///
/// NOTE: This is experimental scaffolding for discussion and is not yet
/// intended for production use.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUBalancingScheduler.h"
#include <cmath>

using namespace llvm;

static cl::opt<double> BalanceOccWeight(
    "amdgpu-balance-occ-weight", cl::Hidden,
    cl::desc("Weight of occupancy in schedule cost (0..1)"), cl::init(0.5));
static cl::opt<double> BalanceIlpWeight(
    "amdgpu-balance-ilp-weight", cl::Hidden,
    cl::desc("Weight of ILP/length metric in schedule cost (0..1)"),
    cl::init(0.5));

static cl::opt<unsigned> BalancedStageMaxInstr(
  "amdgpu-balanced-stage-max-instr", cl::Hidden,
  cl::desc("Max instructions per function to run BalancedReschedule stage;"
       " 0 means no limit"),
  cl::init(1000));

AMDGPUBalancingSchedStrategy::AMDGPUBalancingSchedStrategy(
    const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::OccInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::UnclusteredHighRPReschedule);
  SchedStages.push_back(GCNSchedStageID::ClusteredLowOccupancyReschedule);
  SchedStages.push_back(GCNSchedStageID::BalancedReschedule);
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::MemoryClauseInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
}

bool AMDGPUBalancingSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                                SchedCandidate &TryCand,
                                                SchedBoundary *Zone) const {
  return GCNSchedStrategy::tryCandidate(Cand, TryCand, Zone);
}

const GCNCandidateHeuristic *
AMDGPUBalancingSchedStrategy::getStageHeuristic(GCNSchedStageID Stage) const {
  switch (Stage) {
  case GCNSchedStageID::ILPInitialSchedule:
    return getAMDGPUBalancingHeuristic();
  case GCNSchedStageID::BalancedReschedule:
    return getAMDGPUBalancingHeuristic();
  case GCNSchedStageID::OccInitialSchedule:
    return getAMDGPUPRReduceHeuristic();
  default:
    return nullptr;
  }
}

void AMDGPUBalancingSchedStrategy::onStageFinished(GCNSchedStageID Stage,
                                                   const ScheduleMetrics &M,
                                                   unsigned Occupancy) {
  Results[Stage] = StageResult{true, M, Occupancy};
}

bool AMDGPUBalancingSchedStrategy::isStageWorseThanPrev(
    GCNSchedStageID Stage) const {
  const GCNSchedStageID *const It =
      std::find(SchedStages.begin(), SchedStages.end(), Stage);
  if (It == SchedStages.end() || It == SchedStages.begin())
    return false; // No previous stage to compare.
  GCNSchedStageID Prev = *(It - 1);
  const StageResult *CurrR = getCachedResult(Stage);
  const StageResult *PrevR = getCachedResult(Prev);
  if (!CurrR || !CurrR->Valid || !PrevR || !PrevR->Valid)
    return false;
  double CurrCost = computeWeightedScheduleCost(CurrR->Metrics, CurrR->Occupancy);
  double PrevCost = computeWeightedScheduleCost(PrevR->Metrics, PrevR->Occupancy);
  return CurrCost + 1e-9 < PrevCost; // Revert if strictly worse with tiny eps.
}

bool AMDGPUBalancingSchedStrategy::shouldRevertStage(
    GCNSchedStageID Stage) const {
  return isStageWorseThanPrev(Stage);
}

void AMDGPUBalancingSchedStrategy::onStageSnapshot(
    GCNSchedStageID Stage, const FullScheduleOrder &Order) {
  StageSnapshots[Stage] = Order;
}

const GCNSchedStrategy::FullScheduleOrder *
AMDGPUBalancingSchedStrategy::getSnapshotToRestore(
    GCNSchedStageID Stage) const {
  if (!isStageWorseThanPrev(Stage))
    return nullptr;

  const GCNSchedStageID *const It =
      std::find(SchedStages.begin(), SchedStages.end(), Stage);
  if (It == SchedStages.begin() || It == SchedStages.end())
    return nullptr;
  GCNSchedStageID Prev = *(It - 1);
  auto SnapIt = StageSnapshots.find(Prev);
  return SnapIt == StageSnapshots.end() ? nullptr : &SnapIt->second;
}

bool AMDGPUBalancingSchedStrategy::shouldRunStage(
    GCNSchedStageID Stage) const {
  if (Stage == GCNSchedStageID::BalancedReschedule) {
    if (BalancedStageMaxInstr == 0)
      return true;
    // Rough size-based guard: skip for very large kernels to bound cost.
    unsigned TotalInstr = 0;
    for (const MachineBasicBlock &MBB : *MF)
      TotalInstr += std::distance(MBB.begin(), MBB.end());
    return TotalInstr <= BalancedStageMaxInstr;
  }
  return true;
}

namespace {

class PRReduceHeuristic final : public GCNCandidateHeuristic {
public:
  bool tryCandidate(const GCNSchedStrategy &S,
                    GenericSchedulerBase::SchedCandidate &Cand,
                    GenericSchedulerBase::SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override {
    // Initialize the candidate if needed.
    if (!Cand.isValid()) {
      TryCand.Reason = GenericSchedulerBase::FirstValid;
      return true;
    }

    // Bias PhysReg Defs and copies to their uses and defs respectively.
    if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                   biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand,
                   GenericSchedulerBase::PhysReg))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

  const bool HasZone = Zone && Zone->DAG;
  // For pressure queries we need ScheduleDAGMILive. Use static_cast here
  // because ScheduleDAGMILive doesn't implement LLVM-style RTTI.
  auto *DAGLive = HasZone ? static_cast<ScheduleDAGMILive *>(Zone->DAG)
              : nullptr;
    // Avoid exceeding the target's limit.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                    GenericSchedulerBase::RegExcess, DAGLive->TRI, DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    // Avoid increasing the max critical pressure in the scheduled region.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                    TryCand, Cand, GenericSchedulerBase::RegCritical,
                    DAGLive->TRI, DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    // Avoid increasing the max pressure of the entire region.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax,
                    TryCand, Cand, GenericSchedulerBase::RegMax, DAGLive->TRI,
                    DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    // We only compare a subset of features when comparing nodes between
    // Top and Bottom boundary. Some properties are simply incomparable.
    const bool SameBoundary = Zone != nullptr;
    if (SameBoundary) {
      // For loops that are acyclic path limited, aggressively schedule for
      // latency. Within a single cycle, whenever CurrMOps > 0, allow normal
      // heuristics to take precedence.
      if (Zone->Rem && Zone->Rem->IsAcyclicLatencyLimited &&
          !Zone->getCurrMOps() && tryLatency(TryCand, Cand, *Zone))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Prioritize instructions that read unbuffered resources by stall cycles.
      if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                  Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand,
                  GenericSchedulerBase::Stall))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
    }

    // TODO: Fix clustering preference.

    if (SameBoundary) {
      // Weak edges are for clustering and other constraints.
      if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                  getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand,
                  GenericSchedulerBase::Weak))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
    }

    // Avoid increasing the max pressure of the entire region.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax,
                    TryCand, Cand, GenericSchedulerBase::RegMax, DAGLive->TRI,
                    DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    if (SameBoundary) {
      // Avoid critical resource consumption and balance the schedule.
      TryCand.initResourceDelta(Zone->DAG, Zone->SchedModel);
      if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                  TryCand, Cand, GenericSchedulerBase::ResourceReduce))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
      if (tryGreater(TryCand.ResDelta.DemandedResources,
                     Cand.ResDelta.DemandedResources, TryCand, Cand,
                     GenericSchedulerBase::ResourceDemand))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Avoid serializing long latency dependence chains.
      // For acyclic path limited loops, latency was already checked above.
      if (TryCand.Policy.ReduceLatency &&
          (!Zone->Rem || !Zone->Rem->IsAcyclicLatencyLimited) &&
          tryLatency(TryCand, Cand, *Zone))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Fall through to original instruction order.
      if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
          (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
        TryCand.Reason = GenericSchedulerBase::NodeOrder;
        return true;
      }
    }

    return false;
  }
};

class LatencyReduceHeuristic final : public GCNCandidateHeuristic {
public:
  bool tryCandidate(const GCNSchedStrategy &S,
                    GenericSchedulerBase::SchedCandidate &Cand,
                    GenericSchedulerBase::SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override {
                            // Initialize the candidate if needed.
    if (!Cand.isValid()) {
      TryCand.Reason = GenericSchedulerBase::FirstValid;
      return true;
    }

    // We only compare a subset of features when comparing nodes between
    // Top and Bottom boundary. Some properties are simply incomparable.
    const bool SameBoundary = Zone != nullptr;
    if (SameBoundary) {
      // For loops that are acyclic path limited, aggressively schedule for
      // latency. Within a single cycle, whenever CurrMOps > 0, allow normal
      // heuristics to take precedence.
      if (Zone->Rem && Zone->Rem->IsAcyclicLatencyLimited &&
          !Zone->getCurrMOps() && tryLatency(TryCand, Cand, *Zone))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Prioritize instructions that read unbuffered resources by stall cycles.
      if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                  Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand,
                  GenericSchedulerBase::Stall))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
    }

        // Avoid serializing long latency dependence chains.
      // For acyclic path limited loops, latency was already checked above.
      if (TryCand.Policy.ReduceLatency &&
          (!Zone->Rem || !Zone->Rem->IsAcyclicLatencyLimited) &&
          tryLatency(TryCand, Cand, *Zone))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

  const bool HasZone = Zone && Zone->DAG;
  auto *DAGLive = HasZone ? static_cast<ScheduleDAGMILive *>(Zone->DAG)
              : nullptr;

    if (SameBoundary) {
      // Avoid critical resource consumption and balance the schedule.
      TryCand.initResourceDelta(Zone->DAG, Zone->SchedModel);
      if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                  TryCand, Cand, GenericSchedulerBase::ResourceReduce))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
      if (tryGreater(TryCand.ResDelta.DemandedResources,
                     Cand.ResDelta.DemandedResources, TryCand, Cand,
                     GenericSchedulerBase::ResourceDemand))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Avoid serializing long latency dependence chains.
      // For acyclic path limited loops, latency was already checked above.
      if (TryCand.Policy.ReduceLatency &&
          (!Zone->Rem || !Zone->Rem->IsAcyclicLatencyLimited) &&
          tryLatency(TryCand, Cand, *Zone))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Fall through to original instruction order.
      if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
          (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
        TryCand.Reason = GenericSchedulerBase::NodeOrder;
        return true;
      }
    }

    // Avoid exceeding the target's limit.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                    GenericSchedulerBase::RegExcess, DAGLive->TRI, DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    // Avoid increasing the max critical pressure in the scheduled region.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                    TryCand, Cand, GenericSchedulerBase::RegCritical,
                    DAGLive->TRI, DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    // Avoid increasing the max pressure of the entire region.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax,
                    TryCand, Cand, GenericSchedulerBase::RegMax, DAGLive->TRI,
                    DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;


    return false;
  }
};

class BalancingHeuristic final : public GCNCandidateHeuristic {
public:
  bool tryCandidate(const GCNSchedStrategy &S,
                    GenericSchedulerBase::SchedCandidate &Cand,
                    GenericSchedulerBase::SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override {
    // Initialize the candidate if needed.
    if (!Cand.isValid()) {
      TryCand.Reason = GenericSchedulerBase::FirstValid;
      return true;
    }

    // Bias PhysReg Defs and copies to their uses and defs respectively.
    if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                   biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand,
                   GenericSchedulerBase::PhysReg))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

  const bool HasZone = Zone && Zone->DAG;
  // For pressure queries we need ScheduleDAGMILive. Use static_cast here
  // because ScheduleDAGMILive doesn't implement LLVM-style RTTI.
  auto *DAGLive = HasZone ? static_cast<ScheduleDAGMILive *>(Zone->DAG)
              : nullptr;
    // Avoid exceeding the target's limit.
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                    GenericSchedulerBase::RegExcess, DAGLive->TRI, DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                    TryCand, Cand, GenericSchedulerBase::RegCritical,
                    DAGLive->TRI, DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    // We only compare a subset of features when comparing nodes between
    // Top and Bottom boundary. Some properties are simply incomparable.
    const bool SameBoundary = Zone != nullptr;
    if (SameBoundary) {
      // For loops that are acyclic path limited, aggressively schedule for
      // latency. Within a single cycle, whenever CurrMOps > 0, allow normal
      // heuristics to take precedence.
      if (Zone->Rem && Zone->Rem->IsAcyclicLatencyLimited &&
          !Zone->getCurrMOps() && tryLatency(TryCand, Cand, *Zone))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Prioritize instructions that read unbuffered resources by stall cycles.
      if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                  Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand,
                  GenericSchedulerBase::Stall))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
    }

    // TODO: Fix clustering preference.

    if (SameBoundary) {
      // Weak edges are for clustering and other constraints.
      if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                  getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand,
                  GenericSchedulerBase::Weak))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
    }
    
    if (DAGLive && DAGLive->isTrackingPressure() &&
        tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                    TryCand, Cand, GenericSchedulerBase::RegCritical,
                    DAGLive->TRI, DAGLive->MF))
      return TryCand.Reason != GenericSchedulerBase::NoCand;

    // No RegMax heuristic.

    if (SameBoundary) {
      // Avoid critical resource consumption and balance the schedule.
      TryCand.initResourceDelta(Zone->DAG, Zone->SchedModel);
      if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                  TryCand, Cand, GenericSchedulerBase::ResourceReduce))
        return TryCand.Reason != GenericSchedulerBase::NoCand;
      if (tryGreater(TryCand.ResDelta.DemandedResources,
                     Cand.ResDelta.DemandedResources, TryCand, Cand,
                     GenericSchedulerBase::ResourceDemand))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Avoid serializing long latency dependence chains.
      // For acyclic path limited loops, latency was already checked above.
      if (TryCand.Policy.ReduceLatency &&
          (!Zone->Rem || !Zone->Rem->IsAcyclicLatencyLimited) &&
          tryLatency(TryCand, Cand, *Zone))
        return TryCand.Reason != GenericSchedulerBase::NoCand;

      // Fall through to original instruction order.
      if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
          (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
        TryCand.Reason = GenericSchedulerBase::NodeOrder;
        return true;
      }
    }

    return false;
  }
};

} // end anonymous namespace

const GCNCandidateHeuristic *llvm::getAMDGPUPRReduceHeuristic() {
  static PRReduceHeuristic H;
  return &H;
}

const GCNCandidateHeuristic *llvm::getAMDGPULatencyReduceHeuristic() {
  static LatencyReduceHeuristic H;
  return &H;
}

const GCNCandidateHeuristic *llvm::getAMDGPUBalancingHeuristic() {
  static BalancingHeuristic H;
  return &H;
}

static inline double occUtility(double O, double k=0.35) {
  return 1.0 - std::exp(-k * O);
}

double llvm::computeWeightedScheduleCost(const ScheduleMetrics &M,
                                         unsigned Occupancy) {
  // - Lower metric means fewer bubbles; invert so higher is better.
  // - Higher occupancy is better.
  const double Uocc = occUtility(Occupancy);
  const double Metric = static_cast<double>(M.getMetric());
  const double InvMetric = Metric > 0.0 ? (1.0 / Metric) : 0.0;
  return BalanceOccWeight * Uocc + BalanceIlpWeight * InvMetric;
}
