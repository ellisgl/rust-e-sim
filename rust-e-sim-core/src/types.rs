//! Element parameter types and model definitions.
//!
//! This module defines the data structures used to describe circuit components
//! and their physical parameters (e.g., transistor Gummel-Poon parameters,
//! diode saturation currents).
//!
//! These structs are "pure model" definitions — they carry the physical
//! constants of a component but do not store connectivity (nodes).
//! Connectivity is handled at the `Netlist` and `CompiledNetlist` layers.

use serde::{Serialize, Deserialize};

/// NPN or PNP polarity for a BJT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarity {
    Npn,
    Pnp,
}

/// Gummel-Poon BJT parameters.  See SPICE 3 documentation for the canonical
/// definitions; this struct mirrors the names used in `transistor.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transistor {
    pub polarity: Polarity,
    /// Forward beta (bf).
    pub beta: f64,
    /// Reverse beta (br).  TS default = 1.
    pub br: Option<f64>,
    /// Saturation current (is).
    pub is: f64,
    /// Forward emission coefficient (nf).
    pub nf: f64,
    /// Reverse emission coefficient (nr).  TS default = 1.
    pub nr: Option<f64>,
    /// Forward Early voltage (vaf).
    pub vaf: f64,
    /// Reverse Early voltage (var).  TS default = 100.
    pub var_: Option<f64>,
    /// Forward knee current (ikf).  TS default = 1e9 (effectively off).
    pub ikf: Option<f64>,
    /// Reverse knee current (ikr).  TS default = 1e9.
    pub ikr: Option<f64>,
    /// B-E leakage saturation current (ise).  TS default = is/bf.
    pub ise: Option<f64>,
    /// B-E leakage emission coefficient (ne).  TS default = 1.5.
    pub ne: Option<f64>,
    /// B-C leakage saturation current (isc).  TS default = is/br.
    pub isc: Option<f64>,
    /// B-C leakage emission coefficient (nc).  TS default = 2.
    pub nc: Option<f64>,
    /// B-E zero-bias junction capacitance.
    pub cje_farads: f64,
    /// B-C zero-bias junction capacitance.
    pub cjc_farads: f64,
    /// Forward transit time.  TS default = 0.
    pub tf_seconds: Option<f64>,
    /// Reverse transit time.  TS default = 0.
    pub tr_seconds: Option<f64>,
}

impl Transistor {
    /// Construct a minimal-parameter NPN — useful for tests and a base for
    /// other configurations via struct update syntax.
    pub fn npn_basic(is: f64, beta: f64, nf: f64, vaf: f64) -> Self {
        Self {
            polarity: Polarity::Npn,
            beta,
            br: None,
            is,
            nf,
            nr: None,
            vaf,
            var_: None,
            ikf: None,
            ikr: None,
            ise: None,
            ne: None,
            isc: None,
            nc: None,
            cje_farads: 0.0,
            cjc_farads: 0.0,
            tf_seconds: None,
            tr_seconds: None,
        }
    }

    /// Same as `npn_basic` but PNP.  All other parameters mirror.
    pub fn pnp_basic(is: f64, beta: f64, nf: f64, vaf: f64) -> Self {
        let mut t = Self::npn_basic(is, beta, nf, vaf);
        t.polarity = Polarity::Pnp;
        t
    }
}

/// Shockley diode with optional Zener reverse breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Diode {
    /// Saturation current (A).
    pub is: f64,
    /// Emission coefficient.
    pub n: f64,
    /// Reverse breakdown voltage (V).  `Some` for Zeners, `None` for plain
    /// diodes — when `None` the stamp skips the breakdown contribution.
    pub bv: Option<f64>,
    /// Knee current at breakdown (A).  TS default = 1e-3.
    pub ibv: Option<f64>,
}

impl Diode {
    /// Plain Shockley diode with no Zener breakdown.
    pub fn shockley(is: f64, n: f64) -> Self {
        Self { is, n, bv: None, ibv: None }
    }

    /// Zener with reverse breakdown at `bv` volts.
    pub fn zener(is: f64, n: f64, bv: f64) -> Self {
        Self { is, n, bv: Some(bv), ibv: None }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polarity_eq() {
        assert_eq!(Polarity::Npn, Polarity::Npn);
        assert_ne!(Polarity::Npn, Polarity::Pnp);
    }

    #[test]
    fn transistor_basic_constructors() {
        let q = Transistor::npn_basic(1e-14, 200.0, 1.0, 100.0);
        assert_eq!(q.polarity, Polarity::Npn);
        assert_eq!(q.is, 1e-14);
        assert_eq!(q.beta, 200.0);
        assert!(q.br.is_none());

        let p = Transistor::pnp_basic(1e-14, 50.0, 1.0, 50.0);
        assert_eq!(p.polarity, Polarity::Pnp);
        assert_eq!(p.beta, 50.0);
    }

    #[test]
    fn diode_constructors() {
        let d = Diode::shockley(1e-14, 1.0);
        assert!(d.bv.is_none());
        let z = Diode::zener(1e-14, 1.0, 5.6);
        assert_eq!(z.bv, Some(5.6));
        assert!(z.ibv.is_none()); // default applied by stamp
    }
}
