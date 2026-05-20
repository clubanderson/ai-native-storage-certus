//! IGreeter interface for greeting-oriented components.

use component_macros::define_interface;

/// Provides a greeting prefix for user-facing messages.
///
/// # Examples
///
/// ```no_run
/// use interfaces::IGreeter;
///
/// fn greet(greeter: &dyn IGreeter) -> String {
///     format!("{}, world!", greeter.greeting_prefix())
/// }
/// ```
define_interface! {
    pub IGreeter {
        /// Return the prefix used when building greeting messages.
        fn greeting_prefix(&self) -> &str;
    }
}
