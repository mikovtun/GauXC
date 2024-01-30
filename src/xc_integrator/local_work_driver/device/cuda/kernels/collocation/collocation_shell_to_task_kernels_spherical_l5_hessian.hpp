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


__global__ __launch_bounds__(512,2) void collocation_device_shell_to_task_kernel_spherical_hessian_5(
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
    auto* __restrict__ basis_x_eval = task->dbfx + shoff;
    auto* __restrict__ basis_y_eval = task->dbfy + shoff;
    auto* __restrict__ basis_z_eval = task->dbfz + shoff;

    auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
    auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
    auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
    auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
    auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
    auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;

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
      double radial_eval_alpha = 0.;
      double radial_eval_alpha_squared = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
        radial_eval_alpha += a * e;
        radial_eval_alpha_squared += a * a * e;
      }

      radial_eval_alpha *= -2;
      radial_eval_alpha_squared *= 4;

      

      // Evaluate basis function
      basis_eval[ipt + 0*npts] = 3*sqrt_14*radial_eval*y*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      basis_eval[ipt + 1*npts] = 3*sqrt_35*radial_eval*x*y*z*(x*x - y*y)/2;
      basis_eval[ipt + 2*npts] = sqrt_70*radial_eval*y*(-3*x*x*x*x - 2*x*x*y*y + 24*x*x*z*z + y*y*y*y - 8*y*y*z*z)/16;
      basis_eval[ipt + 3*npts] = sqrt_105*radial_eval*x*y*z*(-x*x - y*y + 2*z*z)/2;
      basis_eval[ipt + 4*npts] = sqrt_15*radial_eval*y*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z)/8;
      basis_eval[ipt + 5*npts] = radial_eval*z*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z)/8;
      basis_eval[ipt + 6*npts] = sqrt_15*radial_eval*x*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z)/8;
      basis_eval[ipt + 7*npts] = sqrt_105*radial_eval*z*(-x*x*x*x + 2*x*x*z*z + y*y*y*y - 2*y*y*z*z)/4;
      basis_eval[ipt + 8*npts] = sqrt_70*radial_eval*x*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z)/16;
      basis_eval[ipt + 9*npts] = 3*sqrt_35*radial_eval*z*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/8;
      basis_eval[ipt + 10*npts] = 3*sqrt_14*radial_eval*x*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = 3*sqrt_14*x*y*(20*radial_eval*(x*x - y*y) + radial_eval_alpha*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y))/16;
      basis_x_eval[ipt + 1*npts] = 3*sqrt_35*y*z*(radial_eval*(3*x*x - y*y) + radial_eval_alpha*x*x*(x*x - y*y))/2;
      basis_x_eval[ipt + 2*npts] = sqrt_70*x*y*(-4*radial_eval*(3*x*x + y*y - 12*z*z) - radial_eval_alpha*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      basis_x_eval[ipt + 3*npts] = sqrt_105*y*z*(-radial_eval*(3*x*x + y*y - 2*z*z) - radial_eval_alpha*x*x*(x*x + y*y - 2*z*z))/2;
      basis_x_eval[ipt + 4*npts] = sqrt_15*x*y*(4*radial_eval*(x*x + y*y - 6*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_x_eval[ipt + 5*npts] = x*z*(20*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      basis_x_eval[ipt + 6*npts] = sqrt_15*(radial_eval*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + 4*x*x*(x*x + y*y - 6*z*z) + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z) + radial_eval_alpha*x*x*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_x_eval[ipt + 7*npts] = sqrt_105*x*z*(-4*radial_eval*(x*x - z*z) - radial_eval_alpha*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      basis_x_eval[ipt + 8*npts] = sqrt_70*(radial_eval*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 4*x*x*(-x*x + y*y + 4*z*z) + 3*y*y*y*y - 24*y*y*z*z) + radial_eval_alpha*x*x*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      basis_x_eval[ipt + 9*npts] = 3*sqrt_35*x*z*(4*radial_eval*(x*x - 3*y*y) + radial_eval_alpha*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      basis_x_eval[ipt + 10*npts] = 3*sqrt_14*(5*radial_eval*x*x*x*x - 30*radial_eval*x*x*y*y + 5*radial_eval*y*y*y*y + radial_eval_alpha*x*x*x*x*x*x - 10*radial_eval_alpha*x*x*x*x*y*y + 5*radial_eval_alpha*x*x*y*y*y*y)/16;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = 3*sqrt_14*(5*radial_eval*x*x*x*x - 30*radial_eval*x*x*y*y + 5*radial_eval*y*y*y*y + 5*radial_eval_alpha*x*x*x*x*y*y - 10*radial_eval_alpha*x*x*y*y*y*y + radial_eval_alpha*y*y*y*y*y*y)/16;
      basis_y_eval[ipt + 1*npts] = 3*sqrt_35*x*z*(-radial_eval*(-x*x + 3*y*y) + radial_eval_alpha*y*y*(x*x - y*y))/2;
      basis_y_eval[ipt + 2*npts] = sqrt_70*(-radial_eval*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z + 4*y*y*(x*x - y*y + 4*z*z)) - radial_eval_alpha*y*y*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      basis_y_eval[ipt + 3*npts] = sqrt_105*x*z*(-radial_eval*(x*x + 3*y*y - 2*z*z) - radial_eval_alpha*y*y*(x*x + y*y - 2*z*z))/2;
      basis_y_eval[ipt + 4*npts] = sqrt_15*(radial_eval*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 4*y*y*(x*x + y*y - 6*z*z) + 8*z*z*z*z) + radial_eval_alpha*y*y*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_y_eval[ipt + 5*npts] = y*z*(20*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      basis_y_eval[ipt + 6*npts] = sqrt_15*x*y*(4*radial_eval*(x*x + y*y - 6*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_y_eval[ipt + 7*npts] = sqrt_105*y*z*(4*radial_eval*(y*y - z*z) - radial_eval_alpha*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      basis_y_eval[ipt + 8*npts] = sqrt_70*x*y*(4*radial_eval*(x*x + 3*y*y - 12*z*z) + radial_eval_alpha*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      basis_y_eval[ipt + 9*npts] = 3*sqrt_35*y*z*(-4*radial_eval*(3*x*x - y*y) + radial_eval_alpha*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      basis_y_eval[ipt + 10*npts] = 3*sqrt_14*x*y*(-20*radial_eval*(x*x - y*y) + radial_eval_alpha*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y))/16;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = 3*sqrt_14*radial_eval_alpha*y*z*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      basis_z_eval[ipt + 1*npts] = 3*sqrt_35*x*y*(radial_eval + radial_eval_alpha*z*z)*(x*x - y*y)/2;
      basis_z_eval[ipt + 2*npts] = sqrt_70*y*z*(16*radial_eval*(3*x*x - y*y) - radial_eval_alpha*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      basis_z_eval[ipt + 3*npts] = sqrt_105*x*y*(radial_eval*(-x*x - y*y + 6*z*z) - radial_eval_alpha*z*z*(x*x + y*y - 2*z*z))/2;
      basis_z_eval[ipt + 4*npts] = sqrt_15*y*z*(-8*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_z_eval[ipt + 5*npts] = radial_eval*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z + z*z*(-80*x*x - 80*y*y + 32*z*z))/8 + radial_eval_alpha*z*z*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z)/8;
      basis_z_eval[ipt + 6*npts] = sqrt_15*x*z*(-8*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_z_eval[ipt + 7*npts] = sqrt_105*(radial_eval*(-x*x*x*x + 2*x*x*z*z + y*y*y*y - 2*y*y*z*z + 4*z*z*(x*x - y*y)) - radial_eval_alpha*z*z*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      basis_z_eval[ipt + 8*npts] = sqrt_70*x*z*(16*radial_eval*(x*x - 3*y*y) + radial_eval_alpha*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      basis_z_eval[ipt + 9*npts] = 3*sqrt_35*(radial_eval + radial_eval_alpha*z*z)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/8;
      basis_z_eval[ipt + 10*npts] = 3*sqrt_14*radial_eval_alpha*x*z*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = 3*sqrt_14*y*(20*radial_eval*(3*x*x - y*y) + 40*radial_eval_alpha*x*x*(x*x - y*y) + (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y))/16;
      basis_xx_eval[ipt + 1*npts] = 3*sqrt_35*x*y*z*(6*radial_eval + 2*radial_eval_alpha*(3*x*x - y*y) + (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(x*x - y*y))/2;
      basis_xx_eval[ipt + 2*npts] = sqrt_70*y*(-4*radial_eval*(9*x*x + y*y - 12*z*z) - 8*radial_eval_alpha*x*x*(3*x*x + y*y - 12*z*z) - (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      basis_xx_eval[ipt + 3*npts] = sqrt_105*x*y*z*(-6*radial_eval - 2*radial_eval_alpha*(3*x*x + y*y - 2*z*z) - (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(x*x + y*y - 2*z*z))/2;
      basis_xx_eval[ipt + 4*npts] = sqrt_15*y*(4*radial_eval*(3*x*x + y*y - 6*z*z) + 8*radial_eval_alpha*x*x*(x*x + y*y - 6*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_xx_eval[ipt + 5*npts] = z*(20*radial_eval*(9*x*x + 3*y*y - 4*z*z) + 40*radial_eval_alpha*x*x*(3*x*x + 3*y*y - 4*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      basis_xx_eval[ipt + 6*npts] = sqrt_15*x*(4*radial_eval*(5*x*x + 3*y*y - 18*z*z) + 2*radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + 4*x*x*(x*x + y*y - 6*z*z) + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_xx_eval[ipt + 7*npts] = sqrt_105*z*(-4*radial_eval*(3*x*x - z*z) - 8*radial_eval_alpha*x*x*(x*x - z*z) - (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      basis_xx_eval[ipt + 8*npts] = sqrt_70*x*(4*radial_eval*(-5*x*x + 3*y*y + 12*z*z) + 2*radial_eval_alpha*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 4*x*x*(-x*x + y*y + 4*z*z) + 3*y*y*y*y - 24*y*y*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      basis_xx_eval[ipt + 9*npts] = 3*sqrt_35*z*(12*radial_eval*(x*x - y*y) + 8*radial_eval_alpha*x*x*(x*x - 3*y*y) + (radial_eval_alpha + radial_eval_alpha_squared*x*x)*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      basis_xx_eval[ipt + 10*npts] = 3*sqrt_14*x*(20*radial_eval*x*x - 60*radial_eval*y*y + 11*radial_eval_alpha*x*x*x*x - 70*radial_eval_alpha*x*x*y*y + 15*radial_eval_alpha*y*y*y*y + radial_eval_alpha_squared*x*x*x*x*x*x - 10*radial_eval_alpha_squared*x*x*x*x*y*y + 5*radial_eval_alpha_squared*x*x*y*y*y*y)/16;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = 3*sqrt_14*x*(20*radial_eval*x*x - 60*radial_eval*y*y + 5*radial_eval_alpha*x*x*x*x - 10*radial_eval_alpha*x*x*y*y - 15*radial_eval_alpha*y*y*y*y + 5*radial_eval_alpha_squared*x*x*x*x*y*y - 10*radial_eval_alpha_squared*x*x*y*y*y*y + radial_eval_alpha_squared*y*y*y*y*y*y)/16;
      basis_xy_eval[ipt + 1*npts] = 3*sqrt_35*z*(3*radial_eval*x*x - 3*radial_eval*y*y + radial_eval_alpha*x*x*x*x - radial_eval_alpha*y*y*y*y + radial_eval_alpha_squared*x*x*x*x*y*y - radial_eval_alpha_squared*x*x*y*y*y*y)/2;
      basis_xy_eval[ipt + 2*npts] = sqrt_70*x*(-12*radial_eval*x*x - 12*radial_eval*y*y + 48*radial_eval*z*z - 3*radial_eval_alpha*x*x*x*x - 18*radial_eval_alpha*x*x*y*y + 24*radial_eval_alpha*x*x*z*z + radial_eval_alpha*y*y*y*y + 24*radial_eval_alpha*y*y*z*z - 3*radial_eval_alpha_squared*x*x*x*x*y*y - 2*radial_eval_alpha_squared*x*x*y*y*y*y + 24*radial_eval_alpha_squared*x*x*y*y*z*z + radial_eval_alpha_squared*y*y*y*y*y*y - 8*radial_eval_alpha_squared*y*y*y*y*z*z)/16;
      basis_xy_eval[ipt + 3*npts] = sqrt_105*z*(-radial_eval*(3*x*x + 3*y*y - 2*z*z) - radial_eval_alpha*x*x*(x*x + 3*y*y - 2*z*z) - radial_eval_alpha*y*y*(3*x*x + y*y - 2*z*z) - radial_eval_alpha_squared*x*x*y*y*(x*x + y*y - 2*z*z))/2;
      basis_xy_eval[ipt + 4*npts] = sqrt_15*x*(4*radial_eval*x*x + 12*radial_eval*y*y - 24*radial_eval*z*z + radial_eval_alpha*x*x*x*x + 10*radial_eval_alpha*x*x*y*y - 12*radial_eval_alpha*x*x*z*z + 9*radial_eval_alpha*y*y*y*y - 60*radial_eval_alpha*y*y*z*z + 8*radial_eval_alpha*z*z*z*z + radial_eval_alpha_squared*x*x*x*x*y*y + 2*radial_eval_alpha_squared*x*x*y*y*y*y - 12*radial_eval_alpha_squared*x*x*y*y*z*z + radial_eval_alpha_squared*y*y*y*y*y*y - 12*radial_eval_alpha_squared*y*y*y*y*z*z + 8*radial_eval_alpha_squared*y*y*z*z*z*z)/8;
      basis_xy_eval[ipt + 5*npts] = x*y*z*(120*radial_eval + 40*radial_eval_alpha*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha_squared*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      basis_xy_eval[ipt + 6*npts] = sqrt_15*y*(12*radial_eval*x*x + 4*radial_eval*y*y - 24*radial_eval*z*z + 9*radial_eval_alpha*x*x*x*x + 10*radial_eval_alpha*x*x*y*y - 60*radial_eval_alpha*x*x*z*z + radial_eval_alpha*y*y*y*y - 12*radial_eval_alpha*y*y*z*z + 8*radial_eval_alpha*z*z*z*z + radial_eval_alpha_squared*x*x*x*x*x*x + 2*radial_eval_alpha_squared*x*x*x*x*y*y - 12*radial_eval_alpha_squared*x*x*x*x*z*z + radial_eval_alpha_squared*x*x*y*y*y*y - 12*radial_eval_alpha_squared*x*x*y*y*z*z + 8*radial_eval_alpha_squared*x*x*z*z*z*z)/8;
      basis_xy_eval[ipt + 7*npts] = sqrt_105*x*y*z*(-4*radial_eval_alpha*x*x + 4*radial_eval_alpha*y*y - radial_eval_alpha_squared*x*x*x*x + 2*radial_eval_alpha_squared*x*x*z*z + radial_eval_alpha_squared*y*y*y*y - 2*radial_eval_alpha_squared*y*y*z*z)/4;
      basis_xy_eval[ipt + 8*npts] = sqrt_70*y*(12*radial_eval*x*x + 12*radial_eval*y*y - 48*radial_eval*z*z - radial_eval_alpha*x*x*x*x + 18*radial_eval_alpha*x*x*y*y - 24*radial_eval_alpha*x*x*z*z + 3*radial_eval_alpha*y*y*y*y - 24*radial_eval_alpha*y*y*z*z - radial_eval_alpha_squared*x*x*x*x*x*x + 2*radial_eval_alpha_squared*x*x*x*x*y*y + 8*radial_eval_alpha_squared*x*x*x*x*z*z + 3*radial_eval_alpha_squared*x*x*y*y*y*y - 24*radial_eval_alpha_squared*x*x*y*y*z*z)/16;
      basis_xy_eval[ipt + 9*npts] = 3*sqrt_35*x*y*z*(-24*radial_eval - 8*radial_eval_alpha*x*x - 8*radial_eval_alpha*y*y + radial_eval_alpha_squared*x*x*x*x - 6*radial_eval_alpha_squared*x*x*y*y + radial_eval_alpha_squared*y*y*y*y)/8;
      basis_xy_eval[ipt + 10*npts] = 3*sqrt_14*y*(-60*radial_eval*x*x + 20*radial_eval*y*y - 15*radial_eval_alpha*x*x*x*x - 10*radial_eval_alpha*x*x*y*y + 5*radial_eval_alpha*y*y*y*y + radial_eval_alpha_squared*x*x*x*x*x*x - 10*radial_eval_alpha_squared*x*x*x*x*y*y + 5*radial_eval_alpha_squared*x*x*y*y*y*y)/16;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = 3*sqrt_14*x*y*z*(20*radial_eval_alpha*(x*x - y*y) + radial_eval_alpha_squared*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y))/16;
      basis_xz_eval[ipt + 1*npts] = 3*sqrt_35*y*(radial_eval*(3*x*x - y*y) + radial_eval_alpha*x*x*(x*x - y*y) + radial_eval_alpha*z*z*(3*x*x - y*y) + radial_eval_alpha_squared*x*x*z*z*(x*x - y*y))/2;
      basis_xz_eval[ipt + 2*npts] = sqrt_70*x*y*z*(96*radial_eval + 36*radial_eval_alpha*x*x - 20*radial_eval_alpha*y*y + 48*radial_eval_alpha*z*z - 3*radial_eval_alpha_squared*x*x*x*x - 2*radial_eval_alpha_squared*x*x*y*y + 24*radial_eval_alpha_squared*x*x*z*z + radial_eval_alpha_squared*y*y*y*y - 8*radial_eval_alpha_squared*y*y*z*z)/16;
      basis_xz_eval[ipt + 3*npts] = sqrt_105*y*(-radial_eval*(3*x*x + y*y - 6*z*z) + radial_eval_alpha*x*x*(-x*x - y*y + 6*z*z) - radial_eval_alpha*z*z*(3*x*x + y*y - 2*z*z) - radial_eval_alpha_squared*x*x*z*z*(x*x + y*y - 2*z*z))/2;
      basis_xz_eval[ipt + 4*npts] = sqrt_15*x*y*z*(-48*radial_eval - 20*radial_eval_alpha*x*x - 20*radial_eval_alpha*y*y + 8*radial_eval_alpha*z*z + radial_eval_alpha_squared*x*x*x*x + 2*radial_eval_alpha_squared*x*x*y*y - 12*radial_eval_alpha_squared*x*x*z*z + radial_eval_alpha_squared*y*y*y*y - 12*radial_eval_alpha_squared*y*y*z*z + 8*radial_eval_alpha_squared*z*z*z*z)/8;
      basis_xz_eval[ipt + 5*npts] = x*(60*radial_eval*x*x + 60*radial_eval*y*y - 240*radial_eval*z*z + 15*radial_eval_alpha*x*x*x*x + 30*radial_eval_alpha*x*x*y*y - 60*radial_eval_alpha*x*x*z*z + 15*radial_eval_alpha*y*y*y*y - 60*radial_eval_alpha*y*y*z*z - 40*radial_eval_alpha*z*z*z*z + 15*radial_eval_alpha_squared*x*x*x*x*z*z + 30*radial_eval_alpha_squared*x*x*y*y*z*z - 40*radial_eval_alpha_squared*x*x*z*z*z*z + 15*radial_eval_alpha_squared*y*y*y*y*z*z - 40*radial_eval_alpha_squared*y*y*z*z*z*z + 8*radial_eval_alpha_squared*z*z*z*z*z*z)/8;
      basis_xz_eval[ipt + 6*npts] = sqrt_15*z*(-72*radial_eval*x*x - 24*radial_eval*y*y + 32*radial_eval*z*z - 19*radial_eval_alpha*x*x*x*x - 18*radial_eval_alpha*x*x*y*y - 4*radial_eval_alpha*x*x*z*z + radial_eval_alpha*y*y*y*y - 12*radial_eval_alpha*y*y*z*z + 8*radial_eval_alpha*z*z*z*z + radial_eval_alpha_squared*x*x*x*x*x*x + 2*radial_eval_alpha_squared*x*x*x*x*y*y - 12*radial_eval_alpha_squared*x*x*x*x*z*z + radial_eval_alpha_squared*x*x*y*y*y*y - 12*radial_eval_alpha_squared*x*x*y*y*z*z + 8*radial_eval_alpha_squared*x*x*z*z*z*z)/8;
      basis_xz_eval[ipt + 7*npts] = sqrt_105*x*(-4*radial_eval*x*x + 12*radial_eval*z*z - radial_eval_alpha*x*x*x*x + 2*radial_eval_alpha*x*x*z*z + radial_eval_alpha*y*y*y*y - 6*radial_eval_alpha*y*y*z*z + 4*radial_eval_alpha*z*z*z*z - radial_eval_alpha_squared*x*x*x*x*z*z + 2*radial_eval_alpha_squared*x*x*z*z*z*z + radial_eval_alpha_squared*y*y*y*y*z*z - 2*radial_eval_alpha_squared*y*y*z*z*z*z)/4;
      basis_xz_eval[ipt + 8*npts] = sqrt_70*z*(48*radial_eval*x*x - 48*radial_eval*y*y + 11*radial_eval_alpha*x*x*x*x - 42*radial_eval_alpha*x*x*y*y + 24*radial_eval_alpha*x*x*z*z + 3*radial_eval_alpha*y*y*y*y - 24*radial_eval_alpha*y*y*z*z - radial_eval_alpha_squared*x*x*x*x*x*x + 2*radial_eval_alpha_squared*x*x*x*x*y*y + 8*radial_eval_alpha_squared*x*x*x*x*z*z + 3*radial_eval_alpha_squared*x*x*y*y*y*y - 24*radial_eval_alpha_squared*x*x*y*y*z*z)/16;
      basis_xz_eval[ipt + 9*npts] = 3*sqrt_35*x*(4*radial_eval*(x*x - 3*y*y) + 4*radial_eval_alpha*z*z*(x*x - 3*y*y) + radial_eval_alpha*(x*x*x*x - 6*x*x*y*y + y*y*y*y) + radial_eval_alpha_squared*z*z*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      basis_xz_eval[ipt + 10*npts] = 3*sqrt_14*z*(5*radial_eval_alpha*x*x*x*x - 30*radial_eval_alpha*x*x*y*y + 5*radial_eval_alpha*y*y*y*y + radial_eval_alpha_squared*x*x*x*x*x*x - 10*radial_eval_alpha_squared*x*x*x*x*y*y + 5*radial_eval_alpha_squared*x*x*y*y*y*y)/16;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = 3*sqrt_14*y*(-60*radial_eval*x*x + 20*radial_eval*y*y + 15*radial_eval_alpha*x*x*x*x - 70*radial_eval_alpha*x*x*y*y + 11*radial_eval_alpha*y*y*y*y + 5*radial_eval_alpha_squared*x*x*x*x*y*y - 10*radial_eval_alpha_squared*x*x*y*y*y*y + radial_eval_alpha_squared*y*y*y*y*y*y)/16;
      basis_yy_eval[ipt + 1*npts] = 3*sqrt_35*x*y*z*(-6*radial_eval - 2*radial_eval_alpha*(-x*x + 3*y*y) + (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(x*x - y*y))/2;
      basis_yy_eval[ipt + 2*npts] = sqrt_70*y*(-12*radial_eval*x*x + 20*radial_eval*y*y - 48*radial_eval*z*z - 9*radial_eval_alpha*x*x*x*x - 14*radial_eval_alpha*x*x*y*y + 72*radial_eval_alpha*x*x*z*z + 11*radial_eval_alpha*y*y*y*y - 56*radial_eval_alpha*y*y*z*z - 3*radial_eval_alpha_squared*x*x*x*x*y*y - 2*radial_eval_alpha_squared*x*x*y*y*y*y + 24*radial_eval_alpha_squared*x*x*y*y*z*z + radial_eval_alpha_squared*y*y*y*y*y*y - 8*radial_eval_alpha_squared*y*y*y*y*z*z)/16;
      basis_yy_eval[ipt + 3*npts] = sqrt_105*x*y*z*(-6*radial_eval - 2*radial_eval_alpha*(x*x + 3*y*y - 2*z*z) - (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(x*x + y*y - 2*z*z))/2;
      basis_yy_eval[ipt + 4*npts] = sqrt_15*y*(4*radial_eval*(3*x*x + 5*y*y - 18*z*z) + 2*radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 4*y*y*(x*x + y*y - 6*z*z) + 8*z*z*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_yy_eval[ipt + 5*npts] = z*(20*radial_eval*(3*x*x + 9*y*y - 4*z*z) + 40*radial_eval_alpha*y*y*(3*x*x + 3*y*y - 4*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      basis_yy_eval[ipt + 6*npts] = sqrt_15*x*(4*radial_eval*(x*x + 3*y*y - 6*z*z) + 8*radial_eval_alpha*y*y*(x*x + y*y - 6*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_yy_eval[ipt + 7*npts] = sqrt_105*z*(4*radial_eval*(3*y*y - z*z) + 8*radial_eval_alpha*y*y*(y*y - z*z) - (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      basis_yy_eval[ipt + 8*npts] = sqrt_70*x*(4*radial_eval*(x*x + 9*y*y - 12*z*z) + 8*radial_eval_alpha*y*y*(x*x + 3*y*y - 12*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      basis_yy_eval[ipt + 9*npts] = 3*sqrt_35*z*(-12*radial_eval*(x*x - y*y) - 8*radial_eval_alpha*y*y*(3*x*x - y*y) + (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      basis_yy_eval[ipt + 10*npts] = 3*sqrt_14*x*(-20*radial_eval*(x*x - 3*y*y) - 40*radial_eval_alpha*y*y*(x*x - y*y) + (radial_eval_alpha + radial_eval_alpha_squared*y*y)*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y))/16;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = 3*sqrt_14*z*(5*radial_eval_alpha*x*x*x*x - 30*radial_eval_alpha*x*x*y*y + 5*radial_eval_alpha*y*y*y*y + 5*radial_eval_alpha_squared*x*x*x*x*y*y - 10*radial_eval_alpha_squared*x*x*y*y*y*y + radial_eval_alpha_squared*y*y*y*y*y*y)/16;
      basis_yz_eval[ipt + 1*npts] = 3*sqrt_35*x*(-radial_eval*(-x*x + 3*y*y) + radial_eval_alpha*y*y*(x*x - y*y) - radial_eval_alpha*z*z*(-x*x + 3*y*y) + radial_eval_alpha_squared*y*y*z*z*(x*x - y*y))/2;
      basis_yz_eval[ipt + 2*npts] = sqrt_70*z*(48*radial_eval*x*x - 48*radial_eval*y*y - 3*radial_eval_alpha*x*x*x*x + 42*radial_eval_alpha*x*x*y*y + 24*radial_eval_alpha*x*x*z*z - 11*radial_eval_alpha*y*y*y*y - 24*radial_eval_alpha*y*y*z*z - 3*radial_eval_alpha_squared*x*x*x*x*y*y - 2*radial_eval_alpha_squared*x*x*y*y*y*y + 24*radial_eval_alpha_squared*x*x*y*y*z*z + radial_eval_alpha_squared*y*y*y*y*y*y - 8*radial_eval_alpha_squared*y*y*y*y*z*z)/16;
      basis_yz_eval[ipt + 3*npts] = sqrt_105*x*(-radial_eval*(x*x + 3*y*y - 6*z*z) + radial_eval_alpha*y*y*(-x*x - y*y + 6*z*z) - radial_eval_alpha*z*z*(x*x + 3*y*y - 2*z*z) - radial_eval_alpha_squared*y*y*z*z*(x*x + y*y - 2*z*z))/2;
      basis_yz_eval[ipt + 4*npts] = sqrt_15*z*(-24*radial_eval*x*x - 72*radial_eval*y*y + 32*radial_eval*z*z + radial_eval_alpha*x*x*x*x - 18*radial_eval_alpha*x*x*y*y - 12*radial_eval_alpha*x*x*z*z - 19*radial_eval_alpha*y*y*y*y - 4*radial_eval_alpha*y*y*z*z + 8*radial_eval_alpha*z*z*z*z + radial_eval_alpha_squared*x*x*x*x*y*y + 2*radial_eval_alpha_squared*x*x*y*y*y*y - 12*radial_eval_alpha_squared*x*x*y*y*z*z + radial_eval_alpha_squared*y*y*y*y*y*y - 12*radial_eval_alpha_squared*y*y*y*y*z*z + 8*radial_eval_alpha_squared*y*y*z*z*z*z)/8;
      basis_yz_eval[ipt + 5*npts] = y*(60*radial_eval*x*x + 60*radial_eval*y*y - 240*radial_eval*z*z + 15*radial_eval_alpha*x*x*x*x + 30*radial_eval_alpha*x*x*y*y - 60*radial_eval_alpha*x*x*z*z + 15*radial_eval_alpha*y*y*y*y - 60*radial_eval_alpha*y*y*z*z - 40*radial_eval_alpha*z*z*z*z + 15*radial_eval_alpha_squared*x*x*x*x*z*z + 30*radial_eval_alpha_squared*x*x*y*y*z*z - 40*radial_eval_alpha_squared*x*x*z*z*z*z + 15*radial_eval_alpha_squared*y*y*y*y*z*z - 40*radial_eval_alpha_squared*y*y*z*z*z*z + 8*radial_eval_alpha_squared*z*z*z*z*z*z)/8;
      basis_yz_eval[ipt + 6*npts] = sqrt_15*x*y*z*(-48*radial_eval - 20*radial_eval_alpha*x*x - 20*radial_eval_alpha*y*y + 8*radial_eval_alpha*z*z + radial_eval_alpha_squared*x*x*x*x + 2*radial_eval_alpha_squared*x*x*y*y - 12*radial_eval_alpha_squared*x*x*z*z + radial_eval_alpha_squared*y*y*y*y - 12*radial_eval_alpha_squared*y*y*z*z + 8*radial_eval_alpha_squared*z*z*z*z)/8;
      basis_yz_eval[ipt + 7*npts] = sqrt_105*y*(4*radial_eval*y*y - 12*radial_eval*z*z - radial_eval_alpha*x*x*x*x + 6*radial_eval_alpha*x*x*z*z + radial_eval_alpha*y*y*y*y - 2*radial_eval_alpha*y*y*z*z - 4*radial_eval_alpha*z*z*z*z - radial_eval_alpha_squared*x*x*x*x*z*z + 2*radial_eval_alpha_squared*x*x*z*z*z*z + radial_eval_alpha_squared*y*y*y*y*z*z - 2*radial_eval_alpha_squared*y*y*z*z*z*z)/4;
      basis_yz_eval[ipt + 8*npts] = sqrt_70*x*y*z*(-96*radial_eval + 20*radial_eval_alpha*x*x - 36*radial_eval_alpha*y*y - 48*radial_eval_alpha*z*z - radial_eval_alpha_squared*x*x*x*x + 2*radial_eval_alpha_squared*x*x*y*y + 8*radial_eval_alpha_squared*x*x*z*z + 3*radial_eval_alpha_squared*y*y*y*y - 24*radial_eval_alpha_squared*y*y*z*z)/16;
      basis_yz_eval[ipt + 9*npts] = 3*sqrt_35*y*(-4*radial_eval*(3*x*x - y*y) - 4*radial_eval_alpha*z*z*(3*x*x - y*y) + radial_eval_alpha*(x*x*x*x - 6*x*x*y*y + y*y*y*y) + radial_eval_alpha_squared*z*z*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      basis_yz_eval[ipt + 10*npts] = 3*sqrt_14*x*y*z*(-20*radial_eval_alpha*(x*x - y*y) + radial_eval_alpha_squared*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y))/16;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = 3*sqrt_14*y*(radial_eval_alpha + radial_eval_alpha_squared*z*z)*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      basis_zz_eval[ipt + 1*npts] = 3*sqrt_35*x*y*z*(3*radial_eval_alpha + radial_eval_alpha_squared*z*z)*(x*x - y*y)/2;
      basis_zz_eval[ipt + 2*npts] = sqrt_70*y*(16*radial_eval*(3*x*x - y*y) + 32*radial_eval_alpha*z*z*(3*x*x - y*y) - (radial_eval_alpha + radial_eval_alpha_squared*z*z)*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      basis_zz_eval[ipt + 3*npts] = sqrt_105*x*y*z*(12*radial_eval + 2*radial_eval_alpha*(-x*x - y*y + 6*z*z) - (radial_eval_alpha + radial_eval_alpha_squared*z*z)*(x*x + y*y - 2*z*z))/2;
      basis_zz_eval[ipt + 4*npts] = sqrt_15*y*(-24*radial_eval*(x*x + y*y - 4*z*z) - 16*radial_eval_alpha*z*z*(3*x*x + 3*y*y - 4*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*z*z)*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_zz_eval[ipt + 5*npts] = z*(-80*radial_eval*(3*x*x + 3*y*y - 2*z*z) + 2*radial_eval_alpha*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z + z*z*(-80*x*x - 80*y*y + 32*z*z)) + (radial_eval_alpha + radial_eval_alpha_squared*z*z)*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      basis_zz_eval[ipt + 6*npts] = sqrt_15*x*(-24*radial_eval*(x*x + y*y - 4*z*z) - 16*radial_eval_alpha*z*z*(3*x*x + 3*y*y - 4*z*z) + (radial_eval_alpha + radial_eval_alpha_squared*z*z)*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      basis_zz_eval[ipt + 7*npts] = sqrt_105*z*(12*radial_eval*(x*x - y*y) + 2*radial_eval_alpha*(-x*x*x*x + 2*x*x*z*z + y*y*y*y - 2*y*y*z*z + 4*z*z*(x*x - y*y)) - (radial_eval_alpha + radial_eval_alpha_squared*z*z)*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      basis_zz_eval[ipt + 8*npts] = sqrt_70*x*(16*radial_eval*(x*x - 3*y*y) + 32*radial_eval_alpha*z*z*(x*x - 3*y*y) + (radial_eval_alpha + radial_eval_alpha_squared*z*z)*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      basis_zz_eval[ipt + 9*npts] = 3*sqrt_35*z*(3*radial_eval_alpha + radial_eval_alpha_squared*z*z)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/8;
      basis_zz_eval[ipt + 10*npts] = 3*sqrt_14*x*(radial_eval_alpha + radial_eval_alpha_squared*z*z)*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = 3*sqrt_14*radial_eval*y*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      ang_eval_1 = 3*sqrt_35*radial_eval*x*y*z*(x*x - y*y)/2;
      ang_eval_2 = sqrt_70*radial_eval*y*(-3*x*x*x*x - 2*x*x*y*y + 24*x*x*z*z + y*y*y*y - 8*y*y*z*z)/16;
      ang_eval_3 = sqrt_105*radial_eval*x*y*z*(-x*x - y*y + 2*z*z)/2;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = sqrt_15*radial_eval*y*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z)/8;
      ang_eval_1 = radial_eval*z*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z)/8;
      ang_eval_2 = sqrt_15*radial_eval*x*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z)/8;
      ang_eval_3 = sqrt_105*radial_eval*z*(-x*x*x*x + 2*x*x*z*z + y*y*y*y - 2*y*y*z*z)/4;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = sqrt_70*radial_eval*x*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z)/16;
      ang_eval_1 = 3*sqrt_35*radial_eval*z*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/8;
      ang_eval_2 = 3*sqrt_14*radial_eval*x*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = 3*sqrt_14*x*y*(20*radial_eval*(x*x - y*y) + radial_eval_alpha*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y))/16;
      dang_eval_y_0 = 3*sqrt_14*(5*radial_eval*x*x*x*x - 30*radial_eval*x*x*y*y + 5*radial_eval*y*y*y*y + 5*radial_eval_alpha*x*x*x*x*y*y - 10*radial_eval_alpha*x*x*y*y*y*y + radial_eval_alpha*y*y*y*y*y*y)/16;
      dang_eval_z_0 = 3*sqrt_14*radial_eval_alpha*y*z*(5*x*x*x*x - 10*x*x*y*y + y*y*y*y)/16;
      dang_eval_x_1 = 3*sqrt_35*y*z*(radial_eval*(3*x*x - y*y) + radial_eval_alpha*x*x*(x*x - y*y))/2;
      dang_eval_y_1 = 3*sqrt_35*x*z*(-radial_eval*(-x*x + 3*y*y) + radial_eval_alpha*y*y*(x*x - y*y))/2;
      dang_eval_z_1 = 3*sqrt_35*x*y*(radial_eval + radial_eval_alpha*z*z)*(x*x - y*y)/2;
      dang_eval_x_2 = sqrt_70*x*y*(-4*radial_eval*(3*x*x + y*y - 12*z*z) - radial_eval_alpha*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      dang_eval_y_2 = sqrt_70*(-radial_eval*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z + 4*y*y*(x*x - y*y + 4*z*z)) - radial_eval_alpha*y*y*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      dang_eval_z_2 = sqrt_70*y*z*(16*radial_eval*(3*x*x - y*y) - radial_eval_alpha*(3*x*x*x*x + 2*x*x*y*y - 24*x*x*z*z - y*y*y*y + 8*y*y*z*z))/16;
      dang_eval_x_3 = sqrt_105*y*z*(-radial_eval*(3*x*x + y*y - 2*z*z) - radial_eval_alpha*x*x*(x*x + y*y - 2*z*z))/2;
      dang_eval_y_3 = sqrt_105*x*z*(-radial_eval*(x*x + 3*y*y - 2*z*z) - radial_eval_alpha*y*y*(x*x + y*y - 2*z*z))/2;
      dang_eval_z_3 = sqrt_105*x*y*(radial_eval*(-x*x - y*y + 6*z*z) - radial_eval_alpha*z*z*(x*x + y*y - 2*z*z))/2;
      basis_x_eval[ipt + 0*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 0*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 0*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 1*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 1*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 1*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 2*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 2*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 2*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 3*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 3*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 3*npts] = dang_eval_z_3;

      dang_eval_x_0 = sqrt_15*x*y*(4*radial_eval*(x*x + y*y - 6*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_y_0 = sqrt_15*(radial_eval*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 4*y*y*(x*x + y*y - 6*z*z) + 8*z*z*z*z) + radial_eval_alpha*y*y*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_z_0 = sqrt_15*y*z*(-8*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_x_1 = x*z*(20*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_y_1 = y*z*(20*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_z_1 = radial_eval*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z + z*z*(-80*x*x - 80*y*y + 32*z*z))/8 + radial_eval_alpha*z*z*(15*x*x*x*x + 30*x*x*y*y - 40*x*x*z*z + 15*y*y*y*y - 40*y*y*z*z + 8*z*z*z*z)/8;
      dang_eval_x_2 = sqrt_15*(radial_eval*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + 4*x*x*(x*x + y*y - 6*z*z) + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z) + radial_eval_alpha*x*x*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_y_2 = sqrt_15*x*y*(4*radial_eval*(x*x + y*y - 6*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_z_2 = sqrt_15*x*z*(-8*radial_eval*(3*x*x + 3*y*y - 4*z*z) + radial_eval_alpha*(x*x*x*x + 2*x*x*y*y - 12*x*x*z*z + y*y*y*y - 12*y*y*z*z + 8*z*z*z*z))/8;
      dang_eval_x_3 = sqrt_105*x*z*(-4*radial_eval*(x*x - z*z) - radial_eval_alpha*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      dang_eval_y_3 = sqrt_105*y*z*(4*radial_eval*(y*y - z*z) - radial_eval_alpha*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      dang_eval_z_3 = sqrt_105*(radial_eval*(-x*x*x*x + 2*x*x*z*z + y*y*y*y - 2*y*y*z*z + 4*z*z*(x*x - y*y)) - radial_eval_alpha*z*z*(x*x*x*x - 2*x*x*z*z - y*y*y*y + 2*y*y*z*z))/4;
      basis_x_eval[ipt + 4*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 4*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 4*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 5*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 5*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 5*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 6*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 6*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 6*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 7*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 7*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 7*npts] = dang_eval_z_3;

      dang_eval_x_0 = sqrt_70*(radial_eval*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 4*x*x*(-x*x + y*y + 4*z*z) + 3*y*y*y*y - 24*y*y*z*z) + radial_eval_alpha*x*x*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      dang_eval_y_0 = sqrt_70*x*y*(4*radial_eval*(x*x + 3*y*y - 12*z*z) + radial_eval_alpha*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      dang_eval_z_0 = sqrt_70*x*z*(16*radial_eval*(x*x - 3*y*y) + radial_eval_alpha*(-x*x*x*x + 2*x*x*y*y + 8*x*x*z*z + 3*y*y*y*y - 24*y*y*z*z))/16;
      dang_eval_x_1 = 3*sqrt_35*x*z*(4*radial_eval*(x*x - 3*y*y) + radial_eval_alpha*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      dang_eval_y_1 = 3*sqrt_35*y*z*(-4*radial_eval*(3*x*x - y*y) + radial_eval_alpha*(x*x*x*x - 6*x*x*y*y + y*y*y*y))/8;
      dang_eval_z_1 = 3*sqrt_35*(radial_eval + radial_eval_alpha*z*z)*(x*x*x*x - 6*x*x*y*y + y*y*y*y)/8;
      dang_eval_x_2 = 3*sqrt_14*(5*radial_eval*x*x*x*x - 30*radial_eval*x*x*y*y + 5*radial_eval*y*y*y*y + radial_eval_alpha*x*x*x*x*x*x - 10*radial_eval_alpha*x*x*x*x*y*y + 5*radial_eval_alpha*x*x*y*y*y*y)/16;
      dang_eval_y_2 = 3*sqrt_14*x*y*(-20*radial_eval*(x*x - y*y) + radial_eval_alpha*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y))/16;
      dang_eval_z_2 = 3*sqrt_14*radial_eval_alpha*x*z*(x*x*x*x - 10*x*x*y*y + 5*y*y*y*y)/16;
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 9*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 9*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 9*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 10*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 10*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 10*npts] = dang_eval_z_2;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
