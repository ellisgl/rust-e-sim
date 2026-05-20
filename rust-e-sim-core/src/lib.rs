//! High-performance circuit simulation kernels.
//!
//! This crate provides the core numerical engines for the `rust-e-sim` simulator.
//! It implements Modified Nodal Analysis (MNA), sparse LU factorization, and
//! nonlinear transient analysis using Newton-Raphson iteration.
//!
//! Architecture
//! ----------
//! The simulator is built in layers:
//! - `sparse`    — Linear algebra: symbolic and numeric sparse LU factorization,
//!                 Minimum-Degree reordering for fill-in reduction.
//! - `linear`    — Fallback dense Gaussian elimination for singular systems.
//! - `netlist`   — Circuit topology and element definitions.
//! - `compile`   — Static analysis: converts a netlist into a fixed MNA structure
//!                 with pre-computed stamps.
//! - `transient` — The main solver loop: Newton-Raphson iteration with Backward
//!                 Euler and BDF-2 (Gear-2) integration.
//!
//! Numerical Stability
//! -------------------
//! The simulation kernels use several SPICE-style techniques to ensure convergence
//! and stability:
//! - `GMIN` stepping/regularisation for near-singular matrices.
//! - `pnjlim` voltage limiting for diodes and transistors.
//! - Symbolic fill-in propagation during LU factorization.
//! - Dense fallback for sparse factorization failures.
//!
//! This crate is `no_std`-compatible (with `alloc`) as it avoids OS-specific
//! APIs and only requires dynamic memory for sparse structures.

pub mod sparse;
pub mod types;
pub mod diode;
pub mod transistor;
pub mod netlist;
pub mod linear;
pub mod compile;
pub mod transient;
