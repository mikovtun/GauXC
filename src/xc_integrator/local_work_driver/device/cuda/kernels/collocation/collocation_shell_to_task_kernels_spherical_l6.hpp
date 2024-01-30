/**
 * GauXC Copyright (c) 2020-2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once
#include "collocation_device_constants.hpp"
#include "device/xc_device_task.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include "device/common/shell_to_task.hpp"
#include <cassert>

namespace GauXC {


__global__ __launch_bounds__(512,2) void collocation_device_shell_to_task_kernel_spherical_6(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[16][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[16][detail::shell_nprim_max + 1];
  double* my_alpha = alpha[threadIdx.x/32];
  double* my_coeff = coeff[threadIdx.x/32];

  for( auto ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
  const uint32_t ntasks      = shell_to_task[ish].ntask;
  const auto shell           = shell_to_task[ish].shell_device;
  const auto task_idx        = shell_to_task[ish].task_idx_device;
  const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;


  // Load Shell Data into registers / SM
  const uint32_t nprim = shell->nprim();
  const double3 O  = *reinterpret_cast<const double3*>(shell->O_data());

  const int global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
  const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

  // Read in coeffs/exps into SM on first warp
  {
    auto* coeff_gm = shell->coeff_data();
    auto* alpha_gm = shell->alpha_data();
    static_assert( detail::shell_nprim_max == cuda::warp_size );
    const int warp_rank = threadIdx.x % cuda::warp_size;
    my_alpha[warp_rank] = alpha_gm[warp_rank];
    my_coeff[warp_rank] = coeff_gm[warp_rank];
  }

  // Loop over tasks assigned to shells
  // Place each task on a different warp + schedule across blocks
  for( int itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {

    const auto*              task   = device_tasks + task_idx[itask];
    const auto* __restrict__ points_x = task->points_x;
    const auto* __restrict__ points_y = task->points_y;
    const auto* __restrict__ points_z = task->points_z;
    const uint32_t           npts   = task->npts;
    const size_t             shoff  = task_shell_offs[itask] * npts;

    auto* __restrict__ basis_eval = task->bf + shoff;


    // Loop over points in task
    // Assign each point to separate thread within the warp
    #pragma unroll 1
    for( int ipt = threadIdx.x % cuda::warp_size; ipt < npts; ipt += cuda::warp_size ) {
      //const double3 point = points[ipt];
      double3 point;
      point.x = points_x[ipt];
      point.y = points_y[ipt];
      point.z = points_z[ipt];


      const auto x = point.x - O.x;
      const auto y = point.y - O.y;
      const auto z = point.z - O.z;
      const auto rsq = x*x + y*y + z*z;

      // Evaluate radial part of bfn
      double radial_eval = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
      }


      

      // Evaluate basis function
      basis_eval[ipt + 0*npts] = sqrt_462*radial_eval*x*y*(3*x*x*x*x - 10*x*x*y*y + 3*y*y*y*y)/16;
      basis_eval[ipt + 1*npts] = 3*sqrt_154*radial_eval*y*z*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      basis_eval[ipt + 2*npts] = 3*sqrt_7*radial_eval*x*y*(-x*x*x*x + 10*x*x*z*z + y*y*y*y - 10*y*y*z*z)/4;
      basis_eval[ipt + 3*npts] = sqrt_210*radial_eval*y*z*(-9*x*x*x*x - 6*x*x*y*y + 24*x*x*z*z + 3*y*y*y*y - 8*y*y*z*z)/16;
      basis_eval[ipt + 4*npts] = sqrt_210*radial_eval*x*y*(x*x*x*x + 2*x*x*y*y - 16*x*x*z*z + y*y*y*y - 16*y*y*z*z + 16*z*z*z*z)/16;
      basis_eval[ipt + 5*npts] = sqrt_21*radial_eval*y*z*(5*x*x*x*x + 10*x*x*y*y - 20*x*x*z*z + 5*y*y*y*y - 20*y*y*z*z + 8*z*z*z*z)/8;
      basis_eval[ipt + 6*npts] = radial_eval*(-5*x*x*x*x*x*x - 15*x*x*x*x*y*y + 90*x*x*x*x*z*z - 15*x*x*y*y*y*y + 180*x*x*y*y*z*z - 120*x*x*z*z*z*z - 5*y*y*y*y*y*y + 90*y*y*y*y*z*z - 120*y*y*z*z*z*z + 16*z*z*z*z*z*z)/16;
      basis_eval[ipt + 7*npts] = sqrt_21*radial_eval*x*z*(5*x*x*x*x + 10*x*x*y*y - 20*x*x*z*z + 5*y*y*y*y - 20*y*y*z*z + 8*z*z*z*z)/8;
      basis_eval[ipt + 8*npts] = sqrt_210*radial_eval*(x*x*x*x*x*x + x*x*x*x*y*y - 16*x*x*x*x*z*z - x*x*y*y*y*y + 16*x*x*z*z*z*z - y*y*y*y*y*y + 16*y*y*y*y*z*z - 16*y*y*z*z*z*z)/32;
      basis_eval[ipt + 9*npts] = sqrt_210*radial_eval*x*z*(-3*x*x*x*x + 6*x*x*y*y + 8*x*x*z*z + 9*y*y*y*y - 24*y*y*z*z)/16;
      basis_eval[ipt + 10*npts] = 3*sqrt_7*radial_eval*(-x*x*x*x*x*x + 5*x*x*x*x*y*y + 10*x*x*x*x*z*z + 5*x*x*y*y*y*y - 60*x*x*y*y*z*z - y*y*y*y*y*y + 10*y*y*y*y*z*z)/16;
      basis_eval[ipt + 11*npts] = 3*sqrt_154*radial_eval*x*z*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;
      basis_eval[ipt + 12*npts] = sqrt_462*radial_eval*(x*x*x*x*x*x - 15*x*x*x*x*y*y + 15*x*x*y*y*y*y - y*y*y*y*y*y)/32;


    





#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = sqrt_462*radial_eval*x*y*(3*x*x*x*x - 10*x*x*y*y + 3*y*y*y*y)/16;
      ang_eval_1 = 3*sqrt_154*radial_eval*y*z*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      ang_eval_2 = 3*sqrt_7*radial_eval*x*y*(-x*x*x*x + 10*x*x*z*z + y*y*y*y - 10*y*y*z*z)/4;
      ang_eval_3 = sqrt_210*radial_eval*y*z*(-9*x*x*x*x - 6*x*x*y*y + 24*x*x*z*z + 3*y*y*y*y - 8*y*y*z*z)/16;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = sqrt_210*radial_eval*x*y*(x*x*x*x + 2*x*x*y*y - 16*x*x*z*z + y*y*y*y - 16*y*y*z*z + 16*z*z*z*z)/16;
      ang_eval_1 = sqrt_21*radial_eval*y*z*(5*x*x*x*x + 10*x*x*y*y - 20*x*x*z*z + 5*y*y*y*y - 20*y*y*z*z + 8*z*z*z*z)/8;
      ang_eval_2 = radial_eval*(-5*x*x*x*x*x*x - 15*x*x*x*x*y*y + 90*x*x*x*x*z*z - 15*x*x*y*y*y*y + 180*x*x*y*y*z*z - 120*x*x*z*z*z*z - 5*y*y*y*y*y*y + 90*y*y*y*y*z*z - 120*y*y*z*z*z*z + 16*z*z*z*z*z*z)/16;
      ang_eval_3 = sqrt_21*radial_eval*x*z*(5*x*x*x*x + 10*x*x*y*y - 20*x*x*z*z + 5*y*y*y*y - 20*y*y*z*z + 8*z*z*z*z)/8;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = sqrt_210*radial_eval*(x*x*x*x*x*x + x*x*x*x*y*y - 16*x*x*x*x*z*z - x*x*y*y*y*y + 16*x*x*z*z*z*z - y*y*y*y*y*y + 16*y*y*y*y*z*z - 16*y*y*z*z*z*z)/32;
      ang_eval_1 = sqrt_210*radial_eval*x*z*(-3*x*x*x*x + 6*x*x*y*y + 8*x*x*z*z + 9*y*y*y*y - 24*y*y*z*z)/16;
      ang_eval_2 = 3*sqrt_7*radial_eval*(-x*x*x*x*x*x + 5*x*x*x*x*y*y + 10*x*x*x*x*z*z + 5*x*x*y*y*y*y - 60*x*x*y*y*z*z - y*y*y*y*y*y + 10*y*y*y*y*z*z)/16;
      ang_eval_3 = 3*sqrt_154*radial_eval*x*z*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = sqrt_462*radial_eval*(x*x*x*x*x*x - 15*x*x*x*x*y*y + 15*x*x*y*y*y*y - y*y*y*y*y*y)/32;
      basis_eval[ipt + 12*npts] = ang_eval_0;


#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC