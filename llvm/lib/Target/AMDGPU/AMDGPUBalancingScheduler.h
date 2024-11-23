//=== AMDGPUBalancingScheduler.cpp - Objective Balancing Scheduler -*- C++ -*=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Interface for the AMDGPU balanced scheduling prototype.
///
/// This interface exposes hooks for:
///   - Defining and registering multiple scheduling stages with different
///     heuristic priorities.
///   - Evaluating candidate schedules with a configurable cost function.
///   - Selecting the best candidate while allowing occupancy to be a
///     soft, tunable parameter instead of a hard limit.
///
/// The intent is to enable:
///   * More balanced trade-offs between occupancy and ILP.
///   * Workload-specific strategies and external plugin integration.
///   * Easier experimentation with cost function tuning.
///
/// This is not yet PR-ready and is intended as an experimental platform
/// for exploring scheduler improvements.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNBALANCINGSCHEDULER_H
#define LLVM_LIB_TARGET_AMDGPU_GCNBALANCINGSCHEDULER_H

#include "GCNSchedStrategy.h"

namespace llvm {

class AMDGPUBalancingSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

  const GCNCandidateHeuristic *
  getStageHeuristic(GCNSchedStageID Stage) const override;

  void onStageFinished(GCNSchedStageID Stage, const ScheduleMetrics &Metrics,
                       unsigned Occupancy) override;

  bool shouldRevertStage(GCNSchedStageID Stage) const override;

  void onStageSnapshot(GCNSchedStageID Stage,
                       const FullScheduleOrder &Order) override;

  const FullScheduleOrder *
  getSnapshotToRestore(GCNSchedStageID Stage) const override;

  bool shouldRunStage(GCNSchedStageID Stage) const override;

private:
  bool tryPRReduceHeuristic(SchedCandidate &Cand, SchedCandidate &TryCand,
                            SchedBoundary *Zone) const;

  bool tryLatencyReduceHeuristic(SchedCandidate &Cand, SchedCandidate &TryCand,
                                 SchedBoundary *Zone) const;

  bool tryBalancedHeuristic(SchedCandidate &Cand, SchedCandidate &TryCand,
                            SchedBoundary *Zone) const;

public:
  AMDGPUBalancingSchedStrategy(const MachineSchedContext *C);

  // Cached results for each completed stage.
  struct StageResult {
    bool Valid = false;
    ScheduleMetrics Metrics;
    unsigned Occupancy = 0;
  };

  const StageResult *getCachedResult(GCNSchedStageID Stage) const {
    auto It = Results.find(Stage);
    return It == Results.end() ? nullptr : &It->second;
  }

private:
  DenseMap<GCNSchedStageID, StageResult> Results;

  // Store full schedule snapshots per stage so we can restore schedules
  // exactly as they were at stage boundaries.
  DenseMap<GCNSchedStageID, FullScheduleOrder> StageSnapshots;

  // Compare weighted cost between the given stage and the previous one.
  bool isStageWorseThanPrev(GCNSchedStageID Stage) const;
};

// Accessors for per-stage heuristics (singletons). These currently delegate
// to the default behavior and are safe to install without changing results.
const GCNCandidateHeuristic *getAMDGPUPRReduceHeuristic();
const GCNCandidateHeuristic *getAMDGPULatencyReduceHeuristic();
const GCNCandidateHeuristic *getAMDGPUBalancingHeuristic();

// Forward declaration for future schedule cache utilities.
struct AMDGPUScheduleChoice {
  unsigned Occupancy = 0;  // waves per SIMD selected
  ScheduleMetrics Metrics; // schedule length/bubbles summary
  // TODO: store a snapshot/handle to the schedule if needed.
};

// Compute a weighted cost from metrics and occupancy. Higher-level code can
// use this to compare cached schedules at different occupancies.
double computeWeightedScheduleCost(const ScheduleMetrics &M,
                                   unsigned Occupancy);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUBALANCINGSCHEDULER_H
