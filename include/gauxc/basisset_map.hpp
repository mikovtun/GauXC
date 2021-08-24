#pragma once

#include <gauxc/basisset.hpp>
#include <gauxc/molecule.hpp>

namespace GauXC {

class BasisSetMap {

  using ao_range    = std::pair<int32_t,int32_t>;
  using shell_range = std::pair<int32_t, int32_t>;

  //int32_t nbf_;     ///< Number of basis functions
  int32_t nshells_; ///< Number of basis shells

  std::vector<int32_t>      shell_sizes_;       ///< Shell sizes
  std::vector<int32_t>      shell_to_first_ao_; ///< Map from shell index to first basis function of that shell
  std::vector<ao_range>     shell_to_ao_range_; ///< Map from shell index to range of basis functions for that shell
  std::vector<int32_t>      shell_to_center_;
  std::vector<shell_range>  center_to_shell_range_;
 
public:

  /**
   *  @brief Construct a BasisSetMap object from a BasisSet
   *
   *  Generate the maps from shell indices to basis function indices
   */
  template <typename F>
  BasisSetMap( const BasisSet<F>& basis, const Molecule& mol ) :
    nshells_( basis.nshells() ) {	

    shell_sizes_.resize( nshells_ );
    for( int32_t i = 0; i < nshells_; ++i )
      shell_sizes_[i] = basis.at(i).size();

    shell_to_first_ao_.reserve( nshells_ );
    shell_to_ao_range_.reserve( nshells_ );

    size_t st_idx = 0;
    for( const auto& shell : basis ) {
      size_t range_end = st_idx + shell.size();
      shell_to_first_ao_.emplace_back( st_idx );
      shell_to_ao_range_.push_back({ st_idx, range_end });
      st_idx = range_end;
    }

#if 1
    shell_to_center_.resize( nshells_ );
    size_t sh_idx = 0;
    for( const auto& shell : basis ) {
      auto at_pos = std::find_if( mol.begin(), mol.end(), [&](const Atom& at) { 
        return at.x == shell.O()[0] and at.y == shell.O()[1] and at.z == shell.O()[2];
      });
      if( at_pos != mol.end() ) shell_to_center_[sh_idx] = std::distance( mol.begin(), at_pos );
      else shell_to_center_[sh_idx] = -1;
      ++sh_idx;
    }
#else
    // Assume basis is sorted wrt Molecule
    shell_to_center_.reserve( nshells_ );
    int32_t atom_idx = 0;

    auto shell_on_atom( auto& center, auto& atom ) {
      return center[0] == atom.x and center[1] == atom.y and center[2] == atom.z;
    }

    for( const auto& shell : basis ) {
      auto& atom = mol.at(atom_idx);
      auto  on_cur_atom = shell_on_atom( shell.O(), atom );
      if( !on_cur_atom ) atom_idx++;
      shell_to_center_.emplace_back(atom_idx);
    }
#endif

  }



  /// Return the map from shell indicies to starting basis functions (const)
  const auto& shell_to_first_ao() const { return shell_to_first_ao_; }
  
  /// Return the map from shell indicies to starting basis functions (non-const)
  auto& shell_to_first_ao()       { return shell_to_first_ao_; }

  /// Return the map from shell indicies to basis function ranges (const)
  const auto& shell_to_ao_range() const { return shell_to_ao_range_; }
  
  /// Return the map from shell indicies to basis function ranges (non-const)
  auto& shell_to_ao_range() { return shell_to_ao_range_; }

  /// Return container that stores the shell sizes (const)
  const auto& shell_sizes() const { return shell_sizes_; }

  /// Return container that stores the shell sizes (non-const)
  auto& shell_sizes() { return shell_sizes_; }

  const auto& shell_to_center() const { return shell_to_center_; }
  auto& shell_to_center() { return shell_to_center_; }


  /**
   *  @brief Get first basis function index for a specified shell
   *
   *  @param[in] i Shell index
   *  @returns   Basis function index for shell "i"
   */
  auto shell_to_first_ao(int32_t i) const { return shell_to_first_ao_.at(i); }

  /**
   *  @brief Get first basis function range for a specified shell
   *
   *  @param[in] i Shell index
   *  @returns   Basis function range for shell "i"
   */
  auto shell_to_ao_range(int32_t i) const { return shell_to_ao_range_.at(i); }

  /**
   *  @brief Get the size of shell "i"
   *
   *  @param[in] i Shell index
   *  @returns   Size of shell "i"
   */
  auto shell_size(int32_t i) const { return shell_sizes_.at(i); }

  const auto& shell_to_center(int32_t i) const { return shell_to_center_[i]; }
  auto& shell_to_center(int32_t i) { return shell_to_center_[i]; }

}; // class BasisSetMap

} // namespace GauXC
