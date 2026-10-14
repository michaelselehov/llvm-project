//===-- GCNPreRAOptimizations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split register tuple initialization into a single pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This is to allow rematerialization of a value instead of spilling. It is
/// supposed to be done after register coalescer to allow it to do its job and
/// before actual register allocation to allow rematerialization.
///
/// Right now the pass only handles 64 bit SGPRs with immediate initializers,
/// although the same shall be possible with other register classes and
/// instructions if necessary.
///
/// This pass also rewrites a FLAT access to %0 + K to use %1 + (K - C), if
/// %1 = %0 + C is computed earlier in the block and stays live after the
/// access:
///
///   %1:vreg_64 = V_LSHL_ADD_U64_e64 %0, 0, -4
///   GLOBAL_STORE_DWORD %0, %2, -8, 0
/// =>
///   %1:vreg_64 = V_LSHL_ADD_U64_e64 %0, 0, -4
///   GLOBAL_STORE_DWORD %1, %2, -4, 0
///
/// It does so only if the access is the last use of %0, so that %0 and %1 no
/// longer overlap. This typically removes the copy of a loop
/// pointer that is incremented before its last access in the iteration.
///
/// This pass also adds register allocation hints to COPY.
/// The hints will be post-processed by SIRegisterInfo::getRegAllocationHints.
/// When using True16, we often see COPY moving a 16-bit value between a VGPR_32
/// and a VGPR_16. If we use the VGPR_16 that corresponds to the lo16 bits of
/// the VGPR_32, the COPY can be completely eliminated.
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNPreRAAntiHints.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-optimizations"

static cl::opt<bool>
    EnableAntiHints("amdgpu-anti-hints", cl::Hidden,
                    cl::desc("Enable register allocation anti-hints."),
                    cl::init(true));

namespace {

class GCNPreRAOptimizationsImpl {
private:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  TargetSchedModel SchedModel;

  bool processReg(Register Reg);
  void hintTrue16Copy(const MachineInstr &MI);
  bool optimizeBVHStack(MachineInstr &MI);
  std::optional<std::pair<Register, int64_t>>
  findIncrementedBase(MachineInstr &MI, Register Base) const;
  bool reuseIncrementedBase(MachineInstr &MI);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                    "Pre-RA optimizations", false, false)

char GCNPreRAOptimizationsLegacy::ID = 0;

char &llvm::GCNPreRAOptimizationsID = GCNPreRAOptimizationsLegacy::ID;

bool GCNPreRAOptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;
  bool IsAGPRDst = TRI->isAGPRClass(MRI->getRegClass(Reg));

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermdiate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.

      if (!IsAGPRDst)
        return false;

      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          const MachineOperand &DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  // For AGPR reg, check if live intervals need to be updated.
  if (IsAGPRDst) {
    if (Changed) {
      for (Register RegToUpdate : ModifiedRegs) {
        LIS->removeInterval(RegToUpdate);
        LIS->createAndComputeVirtRegInterval(RegToUpdate);
      }
    }

    return Changed;
  }

  // For SGPR reg, check if we can combine instructions.
  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return Changed;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return GCNPreRAOptimizationsImpl(LIS).run(MF);
}

PreservedAnalyses
GCNPreRAOptimizationsPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  GCNPreRAOptimizationsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

void GCNPreRAOptimizationsImpl::hintTrue16Copy(const MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  const TargetRegisterClass *DstRC = TRI->getRegClassForReg(*MRI, Dst);
  bool IsDst16Bit = AMDGPU::VGPR_16RegClass.hasSubClassEq(DstRC);
  if (Dst.isVirtual() && IsDst16Bit && Src.isPhysical() &&
      TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
  if (Src.isVirtual() && MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
      Dst.isPhysical() && DstRC == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
  if (!Dst.isVirtual() || !Src.isVirtual())
    return;
  if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
      MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
    MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
  }
  if (IsDst16Bit && MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
}

bool GCNPreRAOptimizationsImpl::optimizeBVHStack(MachineInstr &MI) {
  SmallVector<Register, 2> UseRegs;

  // Find BVH sources for this DS_BVH_STACK instruction.
  auto CheckUse = [&](MachineOperand &Use) {
    Register Reg = Use.getReg();
    for (const MachineInstr &Src : MRI->def_instructions(Reg)) {
      if (!SIInstrInfo::isImage(Src))
        continue;
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Src.getOpcode());
      const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
          AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
      if (!BaseInfo->BVH)
        continue;
      UseRegs.push_back(Reg);
      break;
    }
  };
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data0));
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data1));

  if (UseRegs.empty())
    return false;

  // Add implicit uses for entire BVH source registers.
  // This avoids partial reallocation of register which could
  // introduce a premature s_wait_bvhcnt.
  for (Register Reg : UseRegs) {
    MI.addOperand(MachineOperand::CreateReg(Reg, false, true));
    LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
  }
  LLVM_DEBUG(dbgs() << "Added implicit uses to: " << MI);

  return true;
}

// Find the closest instruction before \p MI in the same block that reads
// \p Base. If it computes Sum = Base + C as a 64-bit add with an immediate C,
// return Sum and C. No instruction between the add and \p MI may write
// \p Base.
std::optional<std::pair<Register, int64_t>>
GCNPreRAOptimizationsImpl::findIncrementedBase(MachineInstr &MI,
                                               Register Base) const {
  auto IsReg = [&](const MachineOperand *Op, unsigned SubReg) {
    return Op && Op->isReg() && Op->getReg() == Base &&
           Op->getSubReg() == SubReg;
  };
  // Match Src0 + Src1 == Base.SubReg + Imm in either operand order.
  auto GetAddImm = [&](const MachineInstr &Def,
                       unsigned SubReg) -> std::optional<int64_t> {
    const MachineOperand *Src0 =
        TII->getNamedOperand(Def, AMDGPU::OpName::src0);
    const MachineOperand *Src1 =
        TII->getNamedOperand(Def, AMDGPU::OpName::src1);
    if (!Src0 || !Src1)
      return std::nullopt;
    if (IsReg(Src0, SubReg))
      return TII->getImmOrMaterializedImm(*MRI, *Src1);
    if (IsReg(Src1, SubReg))
      return TII->getImmOrMaterializedImm(*MRI, *Src0);
    return std::nullopt;
  };

  MachineBasicBlock::iterator Begin = MI.getParent()->begin();
  for (MachineBasicBlock::iterator I = MI.getIterator(); I != Begin;) {
    MachineInstr &Def = *--I;
    if (Def.isDebugInstr())
      continue;
    if (Def.modifiesRegister(Base, TRI))
      return std::nullopt;
    if (!Def.readsRegister(Base, TRI))
      continue;
    const MachineOperand &Dst = Def.getOperand(0);
    if (!Dst.isReg() || !Dst.isDef() || !Dst.getReg().isVirtual())
      return std::nullopt;
    Register Sum = Dst.getReg();
    switch (Def.getOpcode()) {
    case AMDGPU::V_LSHL_ADD_U64_e64: {
      const MachineOperand *Shift =
          TII->getNamedOperand(Def, AMDGPU::OpName::src1);
      const MachineOperand *Addend =
          TII->getNamedOperand(Def, AMDGPU::OpName::src2);
      if (Dst.getSubReg() ||
          !IsReg(TII->getNamedOperand(Def, AMDGPU::OpName::src0), 0) ||
          !Shift || !Shift->isImm() || Shift->getImm() != 0 || !Addend ||
          !Addend->isImm())
        return std::nullopt;
      return std::pair(Sum, Addend->getImm());
    }
    case AMDGPU::V_ADD_U64_e32:
    case AMDGPU::V_ADD_U64_e64: {
      std::optional<int64_t> Imm = GetAddImm(Def, 0);
      if (Dst.getSubReg() || !Imm)
        return std::nullopt;
      return std::pair(Sum, *Imm);
    }
    case AMDGPU::V_ADDC_U32_e32:
    case AMDGPU::V_ADDC_U32_e64: {
      std::optional<int64_t> Hi = GetAddImm(Def, AMDGPU::sub1);
      if (Dst.getSubReg() != AMDGPU::sub1 || !Hi)
        return std::nullopt;
      // The low half must come just before and produce the carry.
      MachineBasicBlock::iterator J = I;
      while (J != Begin && std::prev(J)->isDebugInstr())
        --J;
      if (J == Begin)
        return std::nullopt;
      MachineInstr &LoDef = *std::prev(J);
      if (LoDef.getOpcode() != (Def.getOpcode() == AMDGPU::V_ADDC_U32_e32
                                    ? AMDGPU::V_ADD_CO_U32_e32
                                    : AMDGPU::V_ADD_CO_U32_e64))
        return std::nullopt;
      const MachineOperand &LoDst = LoDef.getOperand(0);
      std::optional<int64_t> Lo = GetAddImm(LoDef, AMDGPU::sub0);
      if (!LoDst.isReg() || LoDst.getReg() != Sum ||
          LoDst.getSubReg() != AMDGPU::sub0 || !Lo)
        return std::nullopt;
      if (Def.getOpcode() == AMDGPU::V_ADDC_U32_e64) {
        const MachineOperand *CarryIn =
            TII->getNamedOperand(Def, AMDGPU::OpName::src2);
        const MachineOperand *CarryOut =
            TII->getNamedOperand(LoDef, AMDGPU::OpName::sdst);
        if (!CarryIn || !CarryOut || !CarryIn->isReg() || !CarryOut->isReg() ||
            CarryIn->getReg() != CarryOut->getReg())
          return std::nullopt;
      }
      return std::pair(
          Sum, static_cast<int64_t>((static_cast<uint64_t>(*Lo) & 0xffffffff) |
                                    (static_cast<uint64_t>(*Hi) << 32)));
    }
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

bool GCNPreRAOptimizationsImpl::reuseIncrementedBase(MachineInstr &MI) {
  if (!SIInstrInfo::isFLAT(MI) || SIInstrInfo::isFLATScratch(MI) ||
      TII->usesASYNC_CNT(MI) || TII->getNamedOperand(MI, AMDGPU::OpName::saddr))
    return false;
  MachineOperand *Addr = TII->getNamedOperand(MI, AMDGPU::OpName::vaddr);
  MachineOperand *OffsetOp = TII->getNamedOperand(MI, AMDGPU::OpName::offset);
  if (!Addr || !OffsetOp || Addr->getSubReg() || !Addr->getReg().isVirtual())
    return false;
  Register Base = Addr->getReg();

  // MI must be the last use of the base, otherwise the base stays live anyway.
  SlotIndex Idx = LIS->getInstructionIndex(MI).getRegSlot();
  if (!LIS->getInterval(Base).Query(Idx).isKill())
    return false;

  std::optional<std::pair<Register, int64_t>> Found =
      findIncrementedBase(MI, Base);
  if (!Found)
    return false;
  auto [Sum, Inc] = *Found;
  int64_t NewOffset = OffsetOp->getImm() - Inc;

  bool IsGlobal = SIInstrInfo::isFLATGlobal(MI);
  if (!TII->isLegalFLATOffset(NewOffset,
                              IsGlobal ? AMDGPUAS::GLOBAL_ADDRESS
                                       : AMDGPUAS::FLAT_ADDRESS,
                              IsGlobal ? AMDGPU::FlatAddrSpace::FlatGlobal
                                       : AMDGPU::FlatAddrSpace::FLAT))
    return false;

  // Coalescing of 2 and 4 byte accesses needs every address component to be
  // aligned. Do not change the alignment of the base unless the offset
  // becomes 0, in which case the base is the full address.
  if (NewOffset != 0 && MI.hasOneMemOperand() && ST->hasGFX90AInsts()) {
    const MachineMemOperand &MMO = **MI.memoperands_begin();
    LocationSize Size = MMO.getSize();
    if (!MMO.isAtomic() && Size.hasValue() && !Size.isScalable()) {
      int64_t Bytes = Size.getValue().getFixedValue();
      if ((Bytes == 2 || Bytes == 4) && Inc % Bytes != 0)
        return false;
    }
  }

  // Reading the sum at MI must not extend its live range.
  const LiveInterval &SumLI = LIS->getInterval(Sum);
  if (!SumLI.liveAt(Idx) ||
      any_of(SumLI.subranges(),
             [&](const LiveInterval::SubRange &SR) { return !SR.liveAt(Idx); }))
    return false;

  const TargetRegisterClass *RC =
      TII->getRegClass(MI.getDesc(), Addr->getOperandNo());
  if (RC && !MRI->constrainRegClass(Sum, RC))
    return false;

  LLVM_DEBUG(dbgs() << "Reuse incremented base " << printReg(Sum, TRI) << " in "
                    << MI);
  Addr->setReg(Sum);
  Addr->setIsKill(false);
  OffsetOp->setImm(NewOffset);
  LIS->removeInterval(Base);
  LIS->createAndComputeVirtRegInterval(Base);
  return true;
}

bool GCNPreRAOptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  this->ST = &ST;
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  // Anti-hints only steer register allocation, so they do not count as a
  // modification of the function.
  if (EnableAntiHints) {
    SchedModel.init(&ST);
    AMDGPU::HazardContext HCtx{TII, TRI, MRI, LIS, &ST, &SchedModel};
    AMDGPU::applyAntiHintRules(MF, HCtx);
  }

  bool Changed = false;

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if ((RC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC)) &&
        (ST.hasGFX90AInsts() || !TRI->isAGPRClass(RC)))
      continue;

    Changed |= processReg(Reg);
  }

  const bool HasBVHStack = ST.hasBVHDualAndBVH8Insts();
  const bool HasRealTrue16 = ST.useRealTrue16Insts();
  const bool HasFlatOffsets = ST.hasFlatInstOffsets();

  if (!HasRealTrue16 && !HasBVHStack && !HasFlatOffsets)
    return Changed;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (HasFlatOffsets && reuseIncrementedBase(MI)) {
        Changed = true;
        continue;
      }
      // Add RA hints to improve True16 COPY elimination.
      if (HasRealTrue16 && MI.getOpcode() == AMDGPU::COPY) {
        hintTrue16Copy(MI);
        continue;
      }
      // Add implicit uses to avoid early wait on intersect ray instructions.
      if (HasBVHStack &&
          (MI.getOpcode() == AMDGPU::DS_BVH_STACK_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP1_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP2_RTN_B64)) {
        Changed |= optimizeBVHStack(MI);
        continue;
      }
    }
  }

  return Changed;
}
