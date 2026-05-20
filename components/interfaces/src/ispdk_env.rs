//! ISPDKEnv interface for owning the process-wide SPDK environment.

use component_macros::define_interface;

use crate::spdk_types::{SpdkEnvError, VfioDevice};

/// Manages initialization and discovery for the shared SPDK/DPDK runtime.
///
/// # Examples
///
/// ```no_run
/// use interfaces::ISPDKEnv;
///
/// fn boot(env: &dyn ISPDKEnv) -> Result<usize, interfaces::SpdkEnvError> {
///     env.init()?;
///     Ok(env.device_count())
/// }
/// ```
define_interface! {
    pub ISPDKEnv {
        /// Initialize the SPDK/DPDK environment, perform pre-flight checks,
        /// and discover VFIO-attached devices.
        fn init(&self) -> Result<(), SpdkEnvError>;

        /// Return all successfully probed VFIO-attached devices.
        fn devices(&self) -> Vec<VfioDevice>;

        /// Return the number of discovered devices.
        fn device_count(&self) -> usize;

        /// Check whether the SPDK environment has been successfully initialized.
        fn is_initialized(&self) -> bool;
    }
}
