//! ILogger interface for components that expose structured log sinks.

use component_macros::define_interface;

/// Receives log messages at standard severity levels.
///
/// # Examples
///
/// ```no_run
/// use interfaces::ILogger;
///
/// fn log_startup(logger: &dyn ILogger) {
///     logger.info("dispatcher started");
///     logger.debug("warming caches");
/// }
/// ```
define_interface! {
    pub ILogger {
        /// Record an error message.
        fn error(&self, msg: &str);
        /// Record a warning message.
        fn warn(&self, msg: &str);
        /// Record an informational message.
        fn info(&self, msg: &str);
        /// Record a debug message.
        fn debug(&self, msg: &str);
    }
}
