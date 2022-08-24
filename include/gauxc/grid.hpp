#pragma once

#include <memory>
#include <gauxc/types.hpp>

namespace GauXC {

/// A named type pertaining to the size of a radial quadrature
using RadialSize   = detail::NamedType< int64_t, struct RadialSizeType  >;

/// A named type pertaining to the size of an angular quadrature
using AngularSize  = detail::NamedType< int64_t, struct AngularSizeType >;

/// A named type pertaining to the number of grid points in a quadrature batch
using BatchSize    = detail::NamedType< int64_t, struct BatchSizeType   >;

/// A named type pertaining to a scaling factor for a radial quadrature
using RadialScale  = detail::NamedType< double,  struct RadialScaleType >;

namespace detail {
  /**
   *  @brief A class which contains the implementation details of a Grid instance
   */
  class GridImpl;
}

/**
 *  @brief A class to manage a particular spherical (atomic) quadrature
 */
class Grid {

  std::shared_ptr<detail::GridImpl> pimpl_; 
    ///< Implementation details of this particular Grid instance

public:

  Grid() = delete;

  Grid( std::shared_ptr<quadrature_type> q, BatchSize bsz = BatchSize(512) );

  /// Copy a Grid object
  Grid( const Grid& );

  /// Move a Grid object
  Grid( Grid&& ) noexcept;

  /// Copy-assign a Grid object
  Grid& operator=( const Grid& );

  /// Move-assign a Grid object
  Grid& operator=( Grid&& ) noexcept;

  /// Destroy a Grid object
  ~Grid() noexcept;

  /**
   *  @brief Get batcher instance for underlying Grid implementation
   *
   *  Const variant
   *
   *  @returns Batcher instance pertaining to the Grid object
   */
  const batcher_type& batcher() const;

  /**
   *  @brief Get batcher instance for underlying Grid implementation
   *
   *  Non-const variant
   *
   *  @returns Batcher instance pertaining to the Grid object
   */
  batcher_type& batcher();

}; // class Grid

} // namespace GauXC
