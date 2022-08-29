#pragma once
#include <gauxc/runtime_environment/fwd.hpp>
#include <memory>
#ifdef GAUXC_ENABLE_MPI
#include <mpi.h>
#endif

namespace GauXC {

namespace detail {
  class RuntimeEnvironmentImpl;
  #ifdef GAUXC_ENABLE_DEVICE
  DeviceRuntimeEnvironment as_device_runtime( const RuntimeEnvironment& );
  #endif
}

class RuntimeEnvironment {

protected:

#ifdef GAUXC_ENABLE_DEVICE
  friend DeviceRuntimeEnvironment 
    detail::as_device_runtime(const RuntimeEnvironment&); 
#endif

  using pimpl_type = detail::RuntimeEnvironmentImpl;
  using pimpl_ptr_type = std::shared_ptr<pimpl_type>;
  pimpl_ptr_type pimpl_;
  RuntimeEnvironment( pimpl_ptr_type ptr );

public:

  explicit RuntimeEnvironment(GAUXC_MPI_CODE(MPI_Comm comm));
  virtual ~RuntimeEnvironment() noexcept;

  RuntimeEnvironment( const RuntimeEnvironment& );
  RuntimeEnvironment( RuntimeEnvironment&& ) noexcept;

  GAUXC_MPI_CODE(MPI_Comm comm() const;)
  int comm_rank() const;
  int comm_size() const;

  int shared_usage_count() const;

};

#ifdef GAUXC_ENABLE_DEVICE
class DeviceRuntimeEnvironment : public RuntimeEnvironment {

private:

  using parent_type = RuntimeEnvironment;
  friend DeviceRuntimeEnvironment 
    detail::as_device_runtime(const RuntimeEnvironment&); 

  using parent_type::pimpl_type;
  using parent_type::pimpl_ptr_type;
  DeviceRuntimeEnvironment( pimpl_ptr_type ptr );

public:

  DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_Comm comm,) void* mem, 
    size_t mem_sz);
  DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_Comm), double fill_fraction);

  ~DeviceRuntimeEnvironment() noexcept;
  DeviceRuntimeEnvironment( const DeviceRuntimeEnvironment& );
  DeviceRuntimeEnvironment( DeviceRuntimeEnvironment&& ) noexcept;

  void* device_memory() const ;
  size_t device_memory_size() const ;
  bool owns_memory() const;
  DeviceBackend* device_backend() const;

};
#endif

}
