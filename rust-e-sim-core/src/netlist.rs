//! Netlist data structures.
//!
//! This module defines the `Netlist` and its constituent `Element` types.
//! It serves as the bridge between the high-level circuit description and
//! the numerical solver.
//!
//! Elements are represented as a Rust enum, allowing for efficient
//! pattern-matching during the compilation and stamping phases.
//!
//! Connectivity:
//! - Node IDs are `u32` identifiers.
//! - Ground is defined by a specific `ground_node_id` provided during construction.
//! - Elements are connected between nodes; node 0 is commonly used as ground but
//!   not required by the core.

use crate::types::{Diode, Transistor};
use serde::{Serialize, Deserialize};

/// A single circuit element.  Each variant carries the topology nodes it
/// connects to plus any model parameters needed by the stamp function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Element {
    Resistor {
        id: String,
        a: u32,
        b: u32,
        resistance_ohms: f64,
    },
    Capacitor {
        id: String,
        a: u32,
        b: u32,
        capacitance_farads: f64,
        initial_voltage: f64,
    },
    Inductor {
        id: String,
        a: u32,
        b: u32,
        inductance_henry: f64,
        saturation_current_a: Option<f64>,
        /// Coupling group identifier — all inductors sharing the same
        /// non-empty group are mutually coupled with strength `k` from the
        /// corresponding `Coupling` element.  `None` for stand-alone
        /// inductors.
        coupling_group: Option<String>,
        /// Winding direction relative to the group's flux axis: +1 or −1.
        /// Used to set the sign of the mutual-inductance term.  Defaults
        /// to +1 for stand-alone inductors.
        coupling_polarity: i32,
    },
    VoltageSource {
        id: String,
        positive_node: u32,
        negative_node: u32,
        voltage: f64,
    },
    Transistor {
        id: String,
        base: u32,
        collector: u32,
        emitter: u32,
        params: Transistor,
    },
    Diode {
        id: String,
        anode: u32,
        cathode: u32,
        params: Diode,
    },
    /// A coupling element binds a set of inductors into a mutual-inductance
    /// group.  Has no nodes of its own — it just supplies the coupling
    /// coefficient `k` (0..1) for all inductors carrying the matching
    /// `coupling_group` string.  Mutual inductance for each pair (i, j) is
    /// `M = k · sqrt(Li · Lj) · si · sj`.
    Coupling {
        id: String,
        coupling_group: String,
        k: f64,
    },

    /// Electromechanical relay (SPDT — single-pole double-throw).
    ///
    /// Physical model: a coil pulls a contact between two positions based
    /// on coil current with hysteresis.  All five terminals are topology
    /// node IDs.
    ///
    /// - Coil:       a resistor between `coil_positive` and `coil_negative`
    ///               with value `coil_resistance_ohms`.
    /// - Contact:    `common` always connects to the active throw via a
    ///               low resistance (`ron_ohms`) and to the inactive throw
    ///               via a high resistance (`roff_ohms`).  When the relay
    ///               is inactive (de-energised) the NC throw is active.
    ///               When active (energised) the NO throw is active.
    ///
    /// State transitions use coil-current thresholds with hysteresis:
    /// transitions to active when |I_coil| > `on_current`, back to inactive
    /// when |I_coil| < `off_current` (with off_current < on_current).
    /// State is held in TransientState::relay_active and updated once per
    /// step after Newton converges (not per-iteration).
    Relay {
        id: String,
        coil_positive: u32,
        coil_negative: u32,
        common: u32,
        normally_closed: u32,
        normally_open: u32,
        coil_resistance_ohms: f64,
        ron_ohms: f64,
        roff_ohms: f64,
        on_current: f64,
        off_current: f64,
    },
}

impl Element {
    /// Element ID — for diagnostics and source-current reporting.
    pub fn id(&self) -> &str {
        match self {
            Element::Resistor { id, .. }
            | Element::Capacitor { id, .. }
            | Element::Inductor { id, .. }
            | Element::VoltageSource { id, .. }
            | Element::Transistor { id, .. }
            | Element::Diode { id, .. }
            | Element::Coupling { id, .. }
            | Element::Relay { id, .. } => id.as_str(),
        }
    }

    /// All topology nodes this element touches.  Used by the compile path
    /// to build the adjacency graph for Minimum Degree reordering.
    /// `Coupling` returns an empty list — it has no nodes of its own.
    pub fn nodes(&self) -> Vec<u32> {
        match self {
            Element::Resistor { a, b, .. }
            | Element::Capacitor { a, b, .. }
            | Element::Inductor { a, b, .. } => vec![*a, *b],
            Element::VoltageSource { positive_node, negative_node, .. } => {
                vec![*positive_node, *negative_node]
            }
            Element::Transistor { base, collector, emitter, .. } => {
                vec![*base, *collector, *emitter]
            }
            Element::Diode { anode, cathode, .. } => vec![*anode, *cathode],
            Element::Coupling { .. } => Vec::new(),
            Element::Relay {
                coil_positive, coil_negative, common, normally_closed, normally_open, ..
            } => vec![*coil_positive, *coil_negative, *common, *normally_closed, *normally_open],
        }
    }
}

/// A complete netlist: every element plus the node-id chosen as ground.
///
/// The caller is responsible for assigning topology-level node IDs.  Ground
/// is identified by `ground_node_id`; ground is at `0 V` in every solve.
/// All other nodes are "non-ground" and get a compact MNA matrix row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Netlist {
    pub elements: Vec<Element>,
    pub ground_node_id: u32,
}

impl Netlist {
    /// Empty netlist ready to be populated by `push`.
    pub fn new(ground_node_id: u32) -> Self {
        Self { elements: Vec::new(), ground_node_id }
    }

    pub fn push(&mut self, e: Element) -> &mut Self {
        self.elements.push(e);
        self
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_nodes_returns_correct_nodes() {
        let r = Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 1000.0,
        };
        assert_eq!(r.nodes(), vec![1, 2]);

        let q = Element::Transistor {
            id: "Q1".into(), base: 3, collector: 4, emitter: 5,
            params: Transistor::npn_basic(1e-14, 200.0, 1.0, 100.0),
        };
        assert_eq!(q.nodes(), vec![3, 4, 5]);
    }

    #[test]
    fn netlist_builds_correctly() {
        let mut n = Netlist::new(0);
        n.push(Element::Resistor { id: "R1".into(), a: 1, b: 0, resistance_ohms: 1e3 });
        n.push(Element::Capacitor { id: "C1".into(), a: 1, b: 0,
            capacitance_farads: 1e-6, initial_voltage: 0.0 });
        assert_eq!(n.elements.len(), 2);
        assert_eq!(n.ground_node_id, 0);
        assert_eq!(n.elements[0].id(), "R1");
    }
}
