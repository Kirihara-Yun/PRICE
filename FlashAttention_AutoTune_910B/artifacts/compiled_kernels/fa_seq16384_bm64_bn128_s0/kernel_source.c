#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q_handle,  GM_ADDR K_handle,  GM_ADDR V_handle,  GM_ADDR Output_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> Q;
  Q.SetGlobalBuffer((__gm__ half*)Q_handle);
  AscendC::GlobalTensor<half> K;
  K.SetGlobalBuffer((__gm__ half*)K_handle);
  AscendC::GlobalTensor<half> V;
  V.SetGlobalBuffer((__gm__ half*)V_handle);
  AscendC::GlobalTensor<half> Output;
  Output.SetGlobalBuffer((__gm__ half*)Output_handle);
  AscendC::GlobalTensor<float> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ float*)workspace_1_handle);
  AscendC::GlobalTensor<half> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ half*)workspace_2_handle);
  AscendC::GlobalTensor<float> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ float*)workspace_3_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto q_l1 = ascend_l1.GetWithOffset<half>(32768, 0);
  auto k_l1 = ascend_l1.GetWithOffset<half>(32768, 65536);
  auto acc_s_l0c = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto acc_s_l1 = ascend_l1.GetWithOffset<half>(8192, 131072);
  auto v_l1 = ascend_l1.GetWithOffset<half>(32768, 147456);
  auto acc_o_l0c = ascend_l0c.GetWithOffset<float>(16384, 16384);
  auto acc_o = ascend_ub.GetWithOffset<float>(16384, 0);
  auto sumexp = ascend_ub.GetWithOffset<float>(32, 65536);
  auto m_i = ascend_ub.GetWithOffset<float>(32, 65664);
  auto acc_s_ub = ascend_ub.GetWithOffset<float>(4096, 65792);
  auto m_i_prev = ascend_ub.GetWithOffset<float>(32, 82176);
  auto acc_s_ub_ = ascend_ub.GetWithOffset<float>(4096, 82304);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(8192, 98688);
  auto m_i_2d = ascend_ub.GetWithOffset<float>(4096, 106880);
  auto sumexp_i_ub = ascend_ub.GetWithOffset<float>(32, 123264);
  auto acc_s_half = ascend_ub.GetWithOffset<half>(4096, 123392);
  auto acc_o_ub = ascend_ub.GetWithOffset<float>(16384, 82304);
  auto acc_o_half = ascend_ub.GetWithOffset<half>(16384, 65792);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    tl::ascend::copy_gm_to_l1<half, 64, 512>(q_l1[0], Q[(cid * 32768)], 512, 64, 512, true);
    for (int32_t k = 0; k < 128; ++k) {
      for (int32_t n_i = 0; n_i < 2; ++n_i) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(3);
        tl::ascend::copy_gm_to_l1<half, 64, 512>(k_l1[0], K[((((cid / 256) * 8388608) + (k * 65536)) + (n_i * 32768))], 512, 64, 512, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(4);
        tl::ascend::gemm_v0<half, float, 64, 64, 512, false, true, 128>(q_l1[0], k_l1[0], acc_s_l0c[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_1[((cid * 8192) + (n_i * 64))], acc_s_l0c[0], 128, 64, 64, 0);
      }
      AscendC::CrossCoreSetFlag<2, PIPE_FIX>(0);
      AscendC::CrossCoreWaitFlag(1);
      tl::ascend::copy_gm_to_l1<half, 64, 128>(acc_s_l1[0], workspace_2[(cid * 8192)], 128, 64, 128, true);
      for (int32_t n_i_1 = 0; n_i_1 < 2; ++n_i_1) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(1);
        tl::ascend::copy_gm_to_l1<half, 128, 256>(v_l1[0], V[((((cid / 256) * 8388608) + (k * 65536)) + (n_i_1 * 256))], 512, 128, 256, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
        tl::ascend::gemm_v0<half, float, 64, 256, 128, false, false, 128>(acc_s_l1[0], v_l1[0], acc_o_l0c[0], ascend_l0a, ascend_l0b, (bool)1, 256);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 256, 0>(workspace_3[((cid * 32768) + (n_i_1 * 256))], acc_o_l0c[0], 512, 64, 256, 0);
      }
      AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
    }
  }
  if ASCEND_IS_AIV {
    tl::ascend::Fill<float>(acc_o[0], 0.000000e+00f, 16384);
    tl::ascend::Fill<float>(sumexp[0], 0.000000e+00f, 32);
    tl::ascend::Fill<float>(m_i[0], -1073741824, 32);
    for (int32_t k_1 = 0; k_1 < 128; ++k_1) {
      tl::ascend::Fill<float>(acc_s_ub[0], 0.000000e+00f, 4096);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 32>(m_i_prev[0], m_i[0], 32, 1, 1, 32, 1, 1);
      AscendC::CrossCoreWaitFlag(0);
      tl::ascend::copy_gm_to_ub<float, 128, 32>(acc_s_ub_[0], workspace_1[((cid * 8192) + (vid * 4096))], 128, 32, 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(acc_s_ub[0], acc_s_ub[0], acc_s_ub_[0], 4096);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Muls(acc_s_ub[0], acc_s_ub[0], 4.419417e-02f, 4096);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_max<float, 32, 128, -1>(m_i[0], acc_s_ub[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max(m_i[0], m_i[0], m_i_prev[0], 32);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(m_i_prev[0], m_i_prev[0], m_i[0], 32);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(m_i_prev[0], m_i_prev[0], 32);
      tl::ascend::Broadcast<float, 2, 1, false>(m_i_2d[0],m_i[0],tmp_ub[0],(uint32_t[]){32, 128}, (uint32_t[]){32, 1});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(acc_s_ub[0], acc_s_ub[0], m_i_2d[0], 4096);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(acc_s_ub[0], acc_s_ub[0], 4096);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  32,  128,  -1>(sumexp_i_ub[0], acc_s_ub[0], tmp_ub[0], true);
      AscendC::Mul(sumexp[0], sumexp[0], m_i_prev[0], 32);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(sumexp[0], sumexp[0], sumexp_i_ub[0], 32);
      tl::ascend::copy_ub_to_ub<half, float, 4096>(acc_s_half[0], acc_s_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<half, 128, 32>(workspace_2[((cid * 8192) + (vid * 4096))], acc_s_half[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
      for (int32_t h_i = 0; h_i < 32; ++h_i) {
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::PipeBarrier<PIPE_ALL>();
        auto m_i_prev_scalar = m_i_prev.GetValue(h_i);
        AscendC::Muls(acc_o[(h_i * 512)], acc_o[(h_i * 512)], m_i_prev_scalar, 512);
        }
      }
      AscendC::CrossCoreWaitFlag(2);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
      tl::ascend::copy_gm_to_ub<float, 512, 32>(acc_o_ub[0], workspace_3[((cid * 32768) + (vid * 16384))], 512, 32, 512, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(acc_o[0], acc_o[0], acc_o_ub[0], 16384);
    }
    for (int32_t h_i_1 = 0; h_i_1 < 32; ++h_i_1) {
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::PipeBarrier<PIPE_ALL>();
      auto sumexp_scalar = 1.0f / (float)sumexp.GetValue(h_i_1);
      AscendC::Muls(acc_o[(h_i_1 * 512)], acc_o[(h_i_1 * 512)], sumexp_scalar, 512);
    }
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<half, float, 16384>(acc_o_half[0], acc_o[0], 32, 512, 512, 32, 512, 512);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    tl::ascend::copy_ub_to_gm<half, 512, 32>(Output[((cid * 32768) + (vid * 16384))], acc_o_half[0], 512, 32, 512);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* K_handle, uint8_t* V_handle, uint8_t* Output_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<32768, nullptr, stream>>>(Q_handle, K_handle, V_handle, Output_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, fftsAddr);
}
