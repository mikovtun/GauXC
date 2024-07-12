/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once

#include <gauxc/molgrid.hpp>


namespace GauXC {

template <typename PointType, typename WeightType>
class CubicQuadrature :
  public Quadrature<CubicQuadrature<PointType,WeightType>> {

  using base_type = Quadrature<CubicQuadrature<PointType,WeightType>>;

public:

  using point_type       = typename base_type::point_type;
  using weight_type      = typename base_type::weight_type;
  using point_container  = typename base_type::point_container;
  using weight_container = typename base_type::weight_container;

  CubicQuadrature(size_t npts, point_type lo, point_type up):
    base_type( npts, lo, up ) { }

  CubicQuadrature( const UniformTrapezoid& ) = default;
  CubicQuadrature( UniformTrapezoid&& ) noexcept = default;
};



template <typename PointType, typename WeightType>
struct quadrature_traits<
  CubicQuadrature<PointType,WeightType>
> {

  using point_type  = PointType;
  using weight_type = WeightType;

  using point_container  = std::vector< point_type >;
  using weight_container = std::vector< weight_type >;

  inline static constexpr bool bound_inclusive = true;

  inline static std::tuple<point_container,weight_container>
    generate( GauXC::CubicGridSpecification cgs ) {
      // Generate the cube grid
      const auto origin = cgs.origin;
      const auto voxelVecs = cgs.voxelVecs;
      const auto xMax = cgs.voxelDims[0];
      const auto yMax = cgs.voxelDims[1];
      const auto zMax = cgs.voxelDims[2];
      const size_t nPts = xMax * yMax * zMax;
      auto points = std::vector<std::array<double,3>>(nPts);
      auto weights = std::vector<double>(nPts, 1.0);

      size_t idP = 0;
      for( size_t idx = 0; idx < xMax; idx++ ) {
        for( size_t idy = 0; idy < yMax; idy++ ) {
          for( size_t idz = 0; idz < zMax; idz++ ) {
            // x,y,z components
            for( size_t idC = 0; idC < 3; idC++ ) {
              points[idP][idC] = origin[idC] + double(idx) * voxelVecs[0][idC] + 
                                               double(idy) * voxelVecs[1][idC] + 
                                               double(idz) * voxelVecs[2][idC];
            }
            ++idP;
          }
        }
      }
			return std::make_tuple( points, weights );

  }




  double slater_radius_64(AtomicNumber);
  double slater_radius_30(AtomicNumber);
  double clementi_radius_67(AtomicNumber);
  double default_atomic_radius(AtomicNumber);

  RadialScale default_mk_radial_scaling_factor( AtomicNumber );
  RadialScale default_mhl_radial_scaling_factor( AtomicNumber );
  RadialScale default_ta_radial_scaling_factor( AtomicNumber );
  RadialScale default_radial_scaling_factor( RadialQuad, AtomicNumber );

  std::tuple<RadialSize,AngularSize> 
    default_grid_size(AtomicNumber, RadialQuad, AtomicGridSizeDefault); 

  struct MolGridFactory {

    static UnprunedAtomicGridSpecification create_default_unpruned_grid_spec(
      AtomicNumber, RadialQuad, RadialSize, AngularSize
    );

    static UnprunedAtomicGridSpecification create_default_unpruned_grid_spec(
      AtomicNumber, RadialQuad, AtomicGridSizeDefault
    );

    template <typename... Args>
    inline static atomic_grid_variant 
      create_default_pruned_grid_spec( PruningScheme scheme, Args&&... args ) {
      return create_pruned_spec( scheme, 
        create_default_unpruned_grid_spec(std::forward<Args>(args)...)
      );
    }

    template <typename... Args>
    inline static atomic_grid_spec_map create_default_grid_spec_map( 
      const Molecule& mol, PruningScheme scheme, Args&&... args ) {

      atomic_grid_spec_map molmap;
      for( const auto& atom : mol ) 
      if( !molmap.count(atom.Z) ) {
        molmap.emplace( atom.Z, 
          create_default_pruned_grid_spec(scheme, atom.Z, 
            std::forward<Args>(args)...)
        );
      }

      return molmap;
    }

    inline static atomic_grid_map generate_gridmap(
      const atomic_grid_spec_map& gs_map, BatchSize bsz ) {

      atomic_grid_map molmap;
      for( const auto& [key, val] : gs_map ) {
        molmap.emplace( key, AtomicGridFactory::generate_grid(val, bsz) );
      }
      return molmap;

    }
    
    // Special instance for CubeGen
    inline static atomic_grid_map create_default_gridmap( 
      const Molecule& mol, PruningScheme scheme, BatchSize bsz,
      CubicGridSpecification cgs ) {
      // Get first atom and attach whole grid to it. Ignore the others
      const auto& atom = mol[0];
      atomic_grid_map molmap;

      // Generate the cube grid
      const auto origin = cgs.origin;
      const auto voxelVecs = cgs.voxelVecs;
      const auto xMax = cgs.voxelDims[0];
      const auto yMax = cgs.voxelDims[1];
      const auto zMax = cgs.voxelDims[2];
      const size_t nPts = xMax * yMax * zMax;
      auto points = std::vector<std::array<double,3>>(nPts);
      auto weights = std::vector<double>(nPts, 1.0);

      size_t idP = 0;
      for( size_t idx = 0; idx < xMax; idx++ ) {
        for( size_t idy = 0; idy < yMax; idy++ ) {
          for( size_t idz = 0; idz < zMax; idz++ ) {
            // x,y,z components
            for( size_t idC = 0; idC < 3; idC++ ) {
              points[idP][idC] = origin[idC] + double(idx) * voxelVecs[0][idC] + 
                                               double(idy) * voxelVecs[1][idC] + 
                                               double(idz) * voxelVecs[2][idC];
            }
            ++idP;
          }
        }
      }

      // Put grid into quadrature
      using quad_type = IntegratorXX::QuadratureBase< std::vector<std::array<double,3>>, std::vector<double> >;

      auto quadrature = quad_type(std::move(points), std::move(weights));

      Grid cubicGrid ( std::make_shared<quad_type> (quadrature), bsz );
      
      molmap.emplace(atom.Z, cubicGrid);
      return molmap;
    }

    template <typename... Args>
    inline static atomic_grid_map create_default_gridmap( 
      const Molecule& mol, PruningScheme scheme, BatchSize bsz,
      Args&&... args ) {

      return generate_gridmap( create_default_grid_spec_map(mol, scheme, 
        std::forward<Args>(args)...), bsz );

    }

    template <typename... Args>
    inline static MolGrid create_default_molgrid( Args&&... args ) {
      return MolGrid( create_default_gridmap(std::forward<Args>(args)...) );
    }

  };

}

