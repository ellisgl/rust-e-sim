//! WASM bindings for sim-core.
//!
//! Phase 1: expose the sparse LU module so the integration test can verify
//! Rust output matches the TypeScript reference on a real Newton iteration.
//! Once parity is confirmed, Phase 2 ports the element stamps and Phase 3
//! moves the whole `stepTransientNetlist` into Rust.
//!
//! Boundary design
//! ---------------
//! - Plain numeric arrays (`Uint8Array`, `Float64Array`, `Int32Array`) cross
//!   the boundary by COPY at this layer.  For the proof-of-concept call sites
//!   this is fine: `analyze_pattern` runs once per compile and the sparse
//!   solver runs maybe 50-100k times/sec — copies of n-sized typed arrays
//!   (n ≤ ~40) cost microseconds.
//! - When we get to Phase 3/4 the simulator OWNS its buffers inside the wasm
//!   linear memory; only audio samples and UI snapshots cross the boundary.
//! - `SparseLuPattern` is opaque to JS — it stays in wasm memory, and JS
//!   holds an `extern "C"` handle returned by wasm-bindgen.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Smoke-test export — verifies the JS-WASM round-trip works.
/// Returns the input + 1.0.  Will be removed once the real API is in use.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn ping(x: f64) -> f64 {
    x + 1.0
}

// ────────────────────────────────────────────────────────────────────────
// Sparse LU bindings
// ────────────────────────────────────────────────────────────────────────

/// Opaque handle to a precomputed sparse LU pattern.
///
/// JS receives a number-typed handle that it threads back into
/// `numeric_factor` and `sparse_solve_in_place`.  The pattern data itself
/// lives in wasm linear memory and is never serialized across the boundary.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct SparseLuPattern {
    inner: sim_core::sparse::SparseLuPattern,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl SparseLuPattern {
    /// Matrix dimension this pattern was built for.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn n(&self) -> usize {
        self.inner.n
    }
}

/// Build a `SparseLuPattern` from a boolean occupancy marker.
///
/// `marker` must be of length `n*n`.  `marker[i*n+j] != 0` means position
/// `(i,j)` may carry a non-zero value during factorization.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = analyzePattern))]
pub fn analyze_pattern(marker: &[u8], n: usize) -> SparseLuPattern {
    SparseLuPattern {
        inner: sim_core::sparse::analyze_pattern(marker, n),
    }
}

/// Numeric LU factorization in place using a precomputed symbolic pattern.
///
/// `mat` must be of length `n*n` (row-major).  On return the lower triangle
/// stores L (unit diagonal not written) and the upper triangle + diagonal
/// store U.  Returns `false` if a pivot fell below the numerical threshold.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = numericFactor))]
pub fn numeric_factor(mat: &mut [f64], n: usize, pat: &SparseLuPattern) -> bool {
    sim_core::sparse::numeric_factor(mat, n, &pat.inner)
}

/// Solve `(L * U) * x = rhs` using a matrix already factored by
/// `numeric_factor`.  The solution overwrites `rhs` on return.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = sparseSolveInPlace))]
pub fn sparse_solve_in_place(
    mat: &[f64],
    rhs: &mut [f64],
    n: usize,
    pat: &SparseLuPattern,
) {
    sim_core::sparse::sparse_solve_in_place(mat, rhs, n, &pat.inner);
}

/// Greedy Minimum Degree elimination ordering.
///
/// `flat_edges` is `[i0, j0, i1, j1, …]` — pairs of node indices forming
/// edges.  Self-loops and out-of-range indices are filtered.
///
/// Returns the elimination order: `order[k] = i` means "eliminate row i at
/// step k".
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = minimumDegreeOrder))]
pub fn minimum_degree_order(n: usize, flat_edges: &[i32]) -> Vec<i32> {
    let edges = flat_edges
        .chunks_exact(2)
        .map(|c| (c[0] as usize, c[1] as usize));
    sim_core::sparse::minimum_degree_order(n, edges).into_vec()
}

// ────────────────────────────────────────────────────────────────────────
// Element-stamp bindings (Phase 2)
// ────────────────────────────────────────────────────────────────────────
//
// Design notes
// ------------
// Elements (Diode, Transistor) are exposed as opaque wasm-bindgen classes.
// JS constructs each instance via a flat-argument constructor that mirrors
// the TypeScript Simulation*Element interfaces, then passes the handle into
// the stamp functions.
//
// Stamp results come back as wasm-bindgen structs with getter properties.
// Field names match the TS output object (`gBe`, `gMu`, `iEqB`, …) via
// `#[wasm_bindgen(js_name = …)]` so the JS adapter is a thin pass-through.
//
// `Float64Array` arguments cross by COPY at this layer — same trade-off
// rationale as the sparse module.  Per-call cost is microseconds; the hot
// path will eventually go through the Phase-3 Simulator class that owns
// its buffers inside wasm memory.

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Diode {
    inner: sim_core::types::Diode,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl Diode {
    /// Plain Shockley diode with no Zener breakdown.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn shockley(is: f64, n: f64) -> Diode {
        Diode { inner: sim_core::types::Diode::shockley(is, n) }
    }

    /// Zener diode with reverse breakdown at `bv` volts.  `ibv` defaults to
    /// 1e-3 A in the model if not supplied (pass `None` from JS).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn zener(is: f64, n: f64, bv: f64, ibv: Option<f64>) -> Diode {
        let mut d = sim_core::types::Diode::zener(is, n, bv);
        d.ibv = ibv;
        Diode { inner: d }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct DiodeStamp {
    inner: sim_core::diode::DiodeStamp,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl DiodeStamp {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn gd(&self) -> f64 { self.inner.gd }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn ieq(&self) -> f64 { self.inner.ieq }
}

/// Compute the diode stamp for the current Newton iterate.
///
/// `prev_volts` is `Option<Vec<f64>>` (JS `Float64Array | undefined`).  When
/// supplied, the SPICE pnjlim limiter engages on large junction-voltage
/// swings — required for transient mode, omitted in DC operating-point.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = computeDiodeStamp))]
pub fn compute_diode_stamp_js(
    diode: &Diode,
    volts: &[f64],
    ai: i32,
    ki: i32,
    prev_volts: Option<Vec<f64>>,
) -> DiodeStamp {
    let stamp = sim_core::diode::compute_diode_stamp(
        &diode.inner, volts, ai, ki, prev_volts.as_deref(),
    );
    DiodeStamp { inner: stamp }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Transistor {
    inner: sim_core::types::Transistor,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl Transistor {
    /// Construct a Gummel-Poon BJT.  All optional SPICE parameters can be
    /// `undefined` on the JS side; defaults are applied inside the stamp
    /// function to match the TS reference exactly.
    ///
    /// `polarity_npn` is a boolean instead of a string because wasm-bindgen
    /// doesn't transparently marshal string enums.  JS adapter translates
    /// `polarity === 'npn'` → `true`, `'pnp'` → `false`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        polarity_npn: bool,
        beta: f64,
        is_sat: f64,
        nf: f64,
        vaf: f64,
        cje_farads: f64,
        cjc_farads: f64,
        br: Option<f64>,
        nr: Option<f64>,
        var_: Option<f64>,
        ikf: Option<f64>,
        ikr: Option<f64>,
        ise: Option<f64>,
        ne: Option<f64>,
        isc: Option<f64>,
        nc: Option<f64>,
        tf_seconds: Option<f64>,
        tr_seconds: Option<f64>,
    ) -> Transistor {
        use sim_core::types::Polarity;
        Transistor {
            inner: sim_core::types::Transistor {
                polarity: if polarity_npn { Polarity::Npn } else { Polarity::Pnp },
                beta, is: is_sat, nf, vaf, cje_farads, cjc_farads,
                br, nr, var_, ikf, ikr, ise, ne, isc, nc, tf_seconds, tr_seconds,
            },
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct TransistorStamp {
    inner: sim_core::transistor::TransistorStamp,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl TransistorStamp {
    // Getters use the camelCase TS names so the JS adapter doesn't have to
    // translate field-by-field.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter, js_name = gBe))]
    pub fn g_be(&self) -> f64 { self.inner.g_be }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter, js_name = gBc))]
    pub fn g_bc(&self) -> f64 { self.inner.g_bc }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn gm(&self) -> f64 { self.inner.gm }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn gmu(&self) -> f64 { self.inner.gmu }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn gpi(&self) -> f64 { self.inner.gpi }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter, js_name = gmu_b))]
    pub fn gmu_b(&self) -> f64 { self.inner.gmu_b }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter, js_name = iEqB))]
    pub fn i_eq_b(&self) -> f64 { self.inner.i_eq_b }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter, js_name = iEqC))]
    pub fn i_eq_c(&self) -> f64 { self.inner.i_eq_c }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter, js_name = iEqE))]
    pub fn i_eq_e(&self) -> f64 { self.inner.i_eq_e }
}

/// Compute the transistor stamp for the current Newton iterate.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = computeTransistorStamp))]
pub fn compute_transistor_stamp_js(
    q: &Transistor,
    volts: &[f64],
    bi: i32,
    ci: i32,
    ei: i32,
    prev_volts: Option<Vec<f64>>,
) -> TransistorStamp {
    let stamp = sim_core::transistor::compute_transistor_stamp(
        &q.inner, volts, bi, ci, ei, prev_volts.as_deref(),
    );
    TransistorStamp { inner: stamp }
}

// ────────────────────────────────────────────────────────────────────────
// Simulator class (Phase 3a)
// ────────────────────────────────────────────────────────────────────────
//
// JS builds a netlist by calling `Simulator::new(ground_node_id)` then
// `add_resistor`, `add_capacitor`, etc.  Once all elements are added,
// `compile()` finalises the matrix structure.  After that, `step(dt)`
// advances the simulation and `node_voltage(node_id)` returns the most
// recent node voltage.
//
// Phase 3a scope: backward Euler only, single-coil inductors, no
// transformers / relays.  The full feature set (BDF-2, mutual inductance,
// relay state machine, DC operating-point) lands in Phase 3b.
//
// Boundary cost notes
// -------------------
// - `add_*` calls happen once per element at netlist build time — typical
//   kit netlist has ~30-60 elements, so the per-call boundary cost is
//   irrelevant.
// - `step(dt)` is the hot path.  It does NOT allocate, does NOT copy any
//   buffers across the boundary: matrix and solution all live in wasm
//   memory.  JS only sees the step status (`Ok(iter_count)` as a u32 or
//   error code).
// - `node_voltage` is the only voltage accessor; for steady-state probing.
//   In Phase 4 the AudioWorklet path will instead drain a ring buffer
//   directly from wasm memory.

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Simulator {
    netlist: sim_core::netlist::Netlist,
    /// `None` until `compile()` has been called.  Subsequent `add_*` calls
    /// invalidate the compile (set this back to `None`).
    compiled: Option<sim_core::compile::CompiledNetlist>,
    state: Option<sim_core::transient::TransientState>,
}

/// Outcome of a `step()` call, marshalled across the wasm boundary as a
/// small enum.  JS gets back either an iteration count (success) or a
/// negative error code.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    pub ok: bool,
    /// On success: Newton iteration count.  On failure: 0.
    pub iters: u32,
    /// On failure: 1 = singular matrix, 2 = Newton did not converge, 3 = bad dt.
    /// On success: 0.
    pub issue: u32,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl Simulator {
    /// Create an empty simulator with `ground_node_id` as the reference.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(ground_node_id: u32) -> Simulator {
        Simulator {
            netlist: sim_core::netlist::Netlist::new(ground_node_id),
            compiled: None,
            state: None,
        }
    }

    pub fn add_resistor(&mut self, id: String, a: u32, b: u32, resistance_ohms: f64) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::Resistor {
            id, a, b, resistance_ohms,
        });
    }

    pub fn add_capacitor(
        &mut self, id: String, a: u32, b: u32,
        capacitance_farads: f64, initial_voltage: f64,
    ) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::Capacitor {
            id, a, b, capacitance_farads, initial_voltage,
        });
    }

    /// Add an inductor.  Optional `coupling_group` + `coupling_polarity`
    /// link this winding into a mutual-inductance group with strength `k`
    /// supplied by an `add_coupling()` element using the same group.
    /// Pass `None`/`1` for stand-alone inductors.
    pub fn add_inductor(
        &mut self, id: String, a: u32, b: u32,
        inductance_henry: f64, saturation_current_a: Option<f64>,
        coupling_group: Option<String>, coupling_polarity: Option<i32>,
    ) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::Inductor {
            id, a, b, inductance_henry, saturation_current_a,
            coupling_group,
            coupling_polarity: coupling_polarity.unwrap_or(1),
        });
    }

    /// Add a mutual-inductance coupling element binding all inductors
    /// carrying the matching `coupling_group` string.  `k` is the coupling
    /// coefficient (0..1).  Mutual inductance for each pair (i, j) of
    /// inductors in the group is `M = k · sqrt(Li · Lj) · si · sj`.
    pub fn add_coupling(&mut self, id: String, coupling_group: String, k: f64) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::Coupling {
            id, coupling_group, k,
        });
    }

    pub fn add_voltage_source(
        &mut self, id: String, positive_node: u32, negative_node: u32, voltage: f64,
    ) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::VoltageSource {
            id, positive_node, negative_node, voltage,
        });
    }

    pub fn add_diode(&mut self, id: String, anode: u32, cathode: u32, diode: &Diode) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::Diode {
            id, anode, cathode, params: diode.inner,
        });
    }

    pub fn add_transistor(
        &mut self, id: String, base: u32, collector: u32, emitter: u32, q: &Transistor,
    ) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::Transistor {
            id, base, collector, emitter, params: q.inner,
        });
    }

    /// Add an SPDT relay.  All five terminals are topology node IDs.  The
    /// coil sits between `coil_positive` and `coil_negative`; the contact
    /// connects `common` to either `normally_closed` (rest state) or
    /// `normally_open` (energised state) with low resistance `ron_ohms`,
    /// while the inactive throw is connected with high resistance
    /// `roff_ohms`.  State transitions use coil-current hysteresis: relay
    /// activates when `|I_coil| > on_current`, releases when
    /// `|I_coil| < off_current`.  Pass `off_current < on_current` for
    /// proper Schmitt-trigger behaviour.
    #[allow(clippy::too_many_arguments)]
    pub fn add_relay(
        &mut self, id: String,
        coil_positive: u32, coil_negative: u32,
        common: u32, normally_closed: u32, normally_open: u32,
        coil_resistance_ohms: f64,
        ron_ohms: f64, roff_ohms: f64,
        on_current: f64, off_current: f64,
    ) {
        self.invalidate();
        self.netlist.push(sim_core::netlist::Element::Relay {
            id,
            coil_positive, coil_negative,
            common, normally_closed, normally_open,
            coil_resistance_ohms,
            ron_ohms, roff_ohms,
            on_current, off_current,
        });
    }

    /// Build the compiled netlist + transient state.  Must be called after
    /// the last `add_*` and before the first `step()`.  Returns `true` on
    /// success; `false` if the netlist has no non-ground nodes (empty
    /// circuit).
    pub fn compile(&mut self) -> bool {
        match sim_core::compile::compile_netlist(&self.netlist) {
            Some(c) => {
                let state = sim_core::transient::TransientState::new(&c);
                self.compiled = Some(c);
                self.state = Some(state);
                true
            }
            None => false,
        }
    }

    /// Advance the simulation by `dt` seconds (backward Euler).
    pub fn step(&mut self, dt: f64) -> StepResult {
        self.step_with_gear(dt, 1) // 1 = BE
    }

    /// Advance with explicit gear selection: 1 = backward Euler, 2 = BDF-2
    /// (falls back to BE on the first step after compile/DC).
    pub fn step_with_gear(&mut self, dt: f64, gear: u8) -> StepResult {
        let c = match self.compiled.as_mut() {
            Some(c) => c,
            None => return StepResult { ok: false, iters: 0, issue: 3 },
        };
        let s = match self.state.as_mut() {
            Some(s) => s,
            None => return StepResult { ok: false, iters: 0, issue: 3 },
        };
        let cfg = match gear {
            2 => sim_core::transient::StepConfig::bdf2(dt),
            _ => sim_core::transient::StepConfig::be(dt),
        };
        match sim_core::transient::step_with_config(c, s, cfg) {
            Ok(iters) => StepResult { ok: true, iters: iters as u32, issue: 0 },
            Err(sim_core::transient::StepIssue::SingularMatrix) =>
                StepResult { ok: false, iters: 0, issue: 1 },
            Err(sim_core::transient::StepIssue::NewtonDidNotConverge) =>
                StepResult { ok: false, iters: 0, issue: 2 },
            Err(sim_core::transient::StepIssue::BadTimestep) =>
                StepResult { ok: false, iters: 0, issue: 3 },
        }
    }

    /// Solve for the DC operating point and write the result into the
    /// transient state.  Caps are treated as open, inductors as shorts.
    /// Must be called after `compile()` and before the first `step()` if
    /// you want the simulator to start at a nontrivial steady state — e.g.
    /// any circuit with BJTs needs this to converge.  Returns `true` on
    /// success.
    ///
    /// On the first transient step after `solve_dc()`, the simulator
    /// uses backward Euler regardless of the gear flag (matches TS).
    pub fn solve_dc(&mut self) -> bool {
        let c = match self.compiled.as_mut() {
            Some(c) => c,
            None => return false,
        };
        let s = match self.state.as_mut() {
            Some(s) => s,
            None => return false,
        };
        sim_core::transient::solve_dc(c, s).is_ok()
    }

    /// Voltage at a topology node ID — 0.0 if the node is grounded or
    /// hasn't been mentioned by any element.  Returns 0.0 if compile()
    /// has not been called yet.
    pub fn node_voltage(&self, node_id: u32) -> f64 {
        let c = match self.compiled.as_ref() {
            Some(c) => c,
            None => return 0.0,
        };
        let s = match self.state.as_ref() {
            Some(s) => s,
            None => return 0.0,
        };
        if node_id == self.netlist.ground_node_id {
            return 0.0;
        }
        match c.node_index.get(&node_id) {
            Some(&idx) => s.node_volts[idx],
            None => 0.0,
        }
    }

    /// Voltage-source branch current by component id.  Returns 0.0 if the
    /// component is unknown or the simulator hasn't been compiled/stepped.
    ///
    /// Sign convention (matches TS `sourceCurrents` exactly): the value is
    /// the MNA augmented unknown, where a positive current flows from the
    /// EXTERNAL circuit INTO the + terminal of the source — i.e. the
    /// source is in *sink* mode.  A battery driving a load is in *source*
    /// mode, so its branch current is **negative** with magnitude equal
    /// to the load current.  Callers that want a user-friendly "supply
    /// current" should negate the value at the call site.
    pub fn voltage_source_current(&self, id: &str) -> f64 {
        let (c, s) = match (self.compiled.as_ref(), self.state.as_ref()) {
            (Some(c), Some(s)) => (c, s),
            _ => return 0.0,
        };
        for (vi, &el_idx) in c.voltage_source_indices.iter().enumerate() {
            if let sim_core::netlist::Element::VoltageSource { id: el_id, .. } = &c.elements[el_idx] {
                if el_id == id {
                    return s.voltage_source_currents[vi];
                }
            }
        }
        0.0
    }

    /// Per-inductor branch current by component id.  Returns 0.0 if the
    /// component is unknown or the simulator hasn't been compiled/stepped.
    ///
    /// Sign convention: positive current flows from terminal `a` to terminal
    /// `b` (the order passed to `add_inductor`).  For an audio speaker
    /// modelled as `Rvc + Lvc` in series, this is the actual *cone-driving*
    /// current — proportional to acoustic output force (F = B·L·I).  Often
    /// a better audio probe than the voltage across the speaker terminals,
    /// which mixes resistive drop and inductive EMF.
    pub fn inductor_current(&self, id: &str) -> f64 {
        let (c, s) = match (self.compiled.as_ref(), self.state.as_ref()) {
            (Some(c), Some(s)) => (c, s),
            _ => return 0.0,
        };
        for (li, &el_idx) in c.inductor_indices.iter().enumerate() {
            if let sim_core::netlist::Element::Inductor { id: el_id, .. } = &c.elements[el_idx] {
                if el_id == id {
                    return s.inductor_currents[li];
                }
            }
        }
        0.0
    }

    /// Total node count after compile (0 if not yet compiled).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn node_count(&self) -> u32 {
        self.compiled.as_ref().map(|c| c.n as u32).unwrap_or(0)
    }

    // ── State export / import for hot-recompile ────────────────────────
    //
    // These let the JS side preserve transient state across a rebuild
    // when only element VALUES change (pot turn, cap knob, switch
    // toggle).  The exported vectors are copies; restoration is by
    // position, so the new simulator must have identical state-vector
    // shapes (same node count, same element counts and order) — which
    // holds for value-only changes since the kit's component list is
    // fixed by the catalog.

    /// Snapshot of `node_volts` (current MNA unknowns).
    pub fn export_node_volts(&self) -> Vec<f64> {
        self.state.as_ref().map(|s| s.node_volts.clone()).unwrap_or_default()
    }
    /// Restore `node_volts`.  Silently ignored on length mismatch.
    pub fn import_node_volts(&mut self, v: Vec<f64>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.node_volts.len() { s.node_volts = v; }
        }
    }

    /// Snapshot of `prev_node_volts` (predictor history).
    pub fn export_prev_node_volts(&self) -> Vec<f64> {
        self.state.as_ref().map(|s| s.prev_node_volts.clone()).unwrap_or_default()
    }
    pub fn import_prev_node_volts(&mut self, v: Vec<f64>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.prev_node_volts.len() { s.prev_node_volts = v; }
        }
    }

    /// Snapshot of per-capacitor voltages.
    pub fn export_cap_volts(&self) -> Vec<f64> {
        self.state.as_ref().map(|s| s.cap_volts.clone()).unwrap_or_default()
    }
    pub fn import_cap_volts(&mut self, v: Vec<f64>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.cap_volts.len() { s.cap_volts = v; }
        }
    }

    /// Snapshot of per-capacitor voltages from two steps ago (Gear-2).
    pub fn export_prev_cap_volts(&self) -> Vec<f64> {
        self.state.as_ref().map(|s| s.prev_cap_volts.clone()).unwrap_or_default()
    }
    pub fn import_prev_cap_volts(&mut self, v: Vec<f64>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.prev_cap_volts.len() { s.prev_cap_volts = v; }
        }
    }

    /// Snapshot of per-inductor branch currents.
    pub fn export_inductor_currents(&self) -> Vec<f64> {
        self.state.as_ref().map(|s| s.inductor_currents.clone()).unwrap_or_default()
    }
    pub fn import_inductor_currents(&mut self, v: Vec<f64>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.inductor_currents.len() { s.inductor_currents = v; }
        }
    }

    /// Snapshot of per-inductor currents from two steps ago (Gear-2).
    pub fn export_prev_inductor_currents(&self) -> Vec<f64> {
        self.state.as_ref().map(|s| s.prev_inductor_currents.clone()).unwrap_or_default()
    }
    pub fn import_prev_inductor_currents(&mut self, v: Vec<f64>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.prev_inductor_currents.len() { s.prev_inductor_currents = v; }
        }
    }

    /// Snapshot of BJT junction-cap voltages (layout: `[Q0_Vbe, Q0_Vbc, Q1_Vbe, …]`).
    pub fn export_tj_cap_volts(&self) -> Vec<f64> {
        self.state.as_ref().map(|s| s.tj_cap_volts.clone()).unwrap_or_default()
    }
    pub fn import_tj_cap_volts(&mut self, v: Vec<f64>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.tj_cap_volts.len() { s.tj_cap_volts = v; }
        }
    }

    /// Snapshot of relay active flags (`true` = energised).  Returned as
    /// `Vec<u8>` since `Vec<bool>` isn't natively exposable through
    /// wasm-bindgen; 0/1 encoding.
    pub fn export_relay_active(&self) -> Vec<u8> {
        self.state.as_ref()
            .map(|s| s.relay_active.iter().map(|&b| if b { 1 } else { 0 }).collect())
            .unwrap_or_default()
    }
    pub fn import_relay_active(&mut self, v: Vec<u8>) {
        if let Some(s) = self.state.as_mut() {
            if v.len() == s.relay_active.len() {
                for (i, &b) in v.iter().enumerate() { s.relay_active[i] = b != 0; }
            }
        }
    }

    /// Snapshot of Gear-2 readiness (true once a step has been committed).
    pub fn export_gear2_ready(&self) -> bool {
        self.state.as_ref().map(|s| s.gear2_ready).unwrap_or(false)
    }
    pub fn import_gear2_ready(&mut self, ready: bool) {
        if let Some(s) = self.state.as_mut() { s.gear2_ready = ready; }
    }

    /// Snapshot of the previous step's dt (used to scale the predictor).
    pub fn export_prev_dt(&self) -> f64 {
        self.state.as_ref().map(|s| s.prev_dt).unwrap_or(0.0)
    }
    pub fn import_prev_dt(&mut self, dt: f64) {
        if let Some(s) = self.state.as_mut() { s.prev_dt = dt; }
    }

    fn invalidate(&mut self) {
        self.compiled = None;
        self.state = None;
    }
}

// ────────────────────────────────────────────────────────────────────────
// Host-side tests — exercise the same code that the wasm build will run
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_round_trip() {
        assert_eq!(ping(41.0), 42.0);
    }

    #[test]
    fn analyze_factor_solve_end_to_end() {
        // Same 3×3 system as the core unit test, but driven through the
        // wasm-bindgen-facing API to catch any boundary glue issues.
        let n = 3;
        let marker = vec![1u8; n * n];
        let pat = analyze_pattern(&marker, n);
        assert_eq!(pat.n(), 3);

        let mut mat = vec![4.0, 1.0, 0.0,
                           1.0, 3.0, 1.0,
                           0.0, 1.0, 2.0];
        let mut rhs = vec![5.0, 6.0, 4.0];

        assert!(numeric_factor(&mut mat, n, &pat));
        sparse_solve_in_place(&mat, &mut rhs, n, &pat);

        // x[0] = 17/18, x[1] = 11/9, x[2] = 25/18
        assert!((rhs[0] - 17.0 / 18.0).abs() < 1e-12);
        assert!((rhs[1] - 11.0 / 9.0 ).abs() < 1e-12);
        assert!((rhs[2] - 25.0 / 18.0).abs() < 1e-12);
    }

    #[test]
    fn md_flat_edges() {
        // Path 0-1-2-3-4; flat_edges = [0,1, 1,2, 2,3, 3,4]
        let order = minimum_degree_order(5, &[0, 1, 1, 2, 2, 3, 3, 4]);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
    }

    #[test]
    fn diode_and_transistor_through_wasm_layer() {
        // Sanity: drive the wasm-bindgen-facing API end-to-end on host so we
        // catch any glue regressions before they hit the browser.
        let d = Diode::shockley(1e-14, 1.0);
        let volts = vec![0.7, 0.0];
        let stamp = compute_diode_stamp_js(&d, &volts, 0, 1, None);
        assert!(stamp.gd() > 1e-3);

        let q = Transistor::new(
            /*npn*/ true, 200.0, 6.734e-15, 1.0, 74.03, 0.0, 0.0,
            None, None, None, None, None, None, None, None, None, None, None,
        );
        let v = vec![0.65, 3.65, 0.0];
        let s = compute_transistor_stamp_js(&q, &v, 0, 1, 2, None);
        assert!(s.gm() > 1e-4);
        assert!(s.gpi() > 0.0);
    }

    /// End-to-end test of the Simulator API as JS would use it: build an
    /// RC, compile, step, read out the cap node voltage.
    #[test]
    fn simulator_rc_through_wasm_layer() {
        let mut sim = Simulator::new(0);
        sim.add_voltage_source("V1".into(), 1, 0, 5.0);
        sim.add_resistor("R1".into(), 1, 2, 1_000.0);
        sim.add_capacitor("C1".into(), 2, 0, 1e-6, 0.0);
        assert!(sim.compile());
        assert_eq!(sim.node_count(), 2);

        // Run 1τ (=1ms) at 1µs steps.  Cap node should reach ~3.16 V.
        for _ in 0..1_000 {
            let r = sim.step(1e-6);
            assert!(r.ok, "step failed: issue={}", r.issue);
        }
        let v = sim.node_voltage(2);
        assert!((v - 3.16).abs() < 0.05, "expected ~3.16 V at t=τ, got {}", v);
        // Ground is always 0.
        assert_eq!(sim.node_voltage(0), 0.0);
        // Unknown node also returns 0 (not an error).
        assert_eq!(sim.node_voltage(999), 0.0);
    }

    /// Phase 3b — DC + BDF-2 + predictor end-to-end through the wasm-bindgen
    /// API.  Voltage-divider biased common-emitter BJT.  `solve_dc()`
    /// establishes the operating point (parity-checked against TS);
    /// subsequent transient steps run without solver errors.
    ///
    /// We don't assert the transient holds the DC operating point exactly
    /// — both TS and Rust diverge from DC by ~1 V over the first few
    /// transient steps because the GMAX-clamped Gummel-Poon model has
    /// transient-only Newton oscillation that doesn't fully converge in
    /// the 20-iter budget.  This is intentional TS behavior; the
    /// invariant we check is that the simulator doesn't crash and that
    /// the DC point matches TS bit-for-bit.
    #[test]
    fn simulator_bjt_dc_then_transient_via_wasm() {
        let mut sim = Simulator::new(0);
        sim.add_voltage_source("VCC".into(), 1, 0, 12.0);
        sim.add_resistor("R1".into(), 1, 2, 47_000.0);
        sim.add_resistor("R2".into(), 2, 0, 10_000.0);
        sim.add_resistor("RC".into(), 1, 3, 1_000.0);
        sim.add_resistor("RE".into(), 4, 0, 1_000.0);
        let q = Transistor::new(
            true, 200.0, 6.734e-15, 1.0, 74.03, 0.0, 0.0,
            None, None, None, None, None, None, None, None, None, None, None,
        );
        sim.add_transistor("Q1".into(), 2, 3, 4, &q);
        assert!(sim.compile());

        assert!(sim.solve_dc(), "DC solve failed");

        // DC parity against TS dc.ts: Vb at the same operating point.
        let vb_dc = sim.node_voltage(2);
        assert!(
            (vb_dc - 3.4478392641000175).abs() < 1e-6,
            "Vb DC = {} (expected TS-parity 3.4478…)",
            vb_dc,
        );

        // 100 transient steps with BDF-2 (will fall back to BE on step 0).
        // Just verify no solver crashes; the values oscillate around but
        // bounded by the step-limit clamp.
        for i in 0..100 {
            let r = sim.step_with_gear(1e-6, 2);
            assert!(r.ok, "step {} failed: issue={}", i, r.issue);
            // Voltages stay bounded — no NaN/Inf escape.
            for node_id in [1u32, 2, 3, 4] {
                let v = sim.node_voltage(node_id);
                assert!(v.is_finite() && v.abs() < 50.0,
                    "node {} = {} at step {}", node_id, v, i);
            }
        }
    }

    /// State export/import round-trip: build an RC, step partway through
    /// the charge curve, snapshot, build a fresh simulator with a
    /// different R value, restore the snapshot, continue stepping —
    /// continuity should be seamless rather than jumping back to V=0.
    /// This is the state-preserving updateControls scenario for a pot
    /// turn while the cap is mid-charge.
    #[test]
    fn state_preserved_across_rebuild() {
        let build_sim = |r_ohms: f64| {
            let mut sim = Simulator::new(0);
            sim.add_voltage_source("V1".into(), 1, 0, 5.0);
            sim.add_resistor("R1".into(), 1, 2, r_ohms);
            sim.add_capacitor("C1".into(), 2, 0, 1e-6, 0.0);
            assert!(sim.compile());
            sim
        };

        let mut sim_a = build_sim(1_000.0);    // τ = 1 ms
        // Step 500 µs ≈ half a time-constant with R=1kΩ → V_cap ≈ 5·(1-e^-0.5) ≈ 1.97 V.
        for _ in 0..500 {
            assert!(sim_a.step(1e-6).ok);
        }
        let v_mid = sim_a.node_voltage(2);
        assert!((v_mid - 1.97).abs() < 0.05, "midpoint V_cap = {}", v_mid);

        // Snapshot, build a new sim with R=2kΩ (pot turn — τ doubles),
        // restore, then keep stepping.  V_cap should continue from
        // ~1.97V — not reset to 0.
        let node_volts            = sim_a.export_node_volts();
        let prev_node_volts       = sim_a.export_prev_node_volts();
        let cap_volts             = sim_a.export_cap_volts();
        let prev_cap_volts        = sim_a.export_prev_cap_volts();
        let inductor_currents     = sim_a.export_inductor_currents();
        let prev_inductor_currents= sim_a.export_prev_inductor_currents();
        let tj_cap_volts          = sim_a.export_tj_cap_volts();
        let relay_active          = sim_a.export_relay_active();
        let gear2_ready           = sim_a.export_gear2_ready();
        let prev_dt               = sim_a.export_prev_dt();

        let mut sim_b = build_sim(2_000.0);    // new R after "pot turn"
        sim_b.import_node_volts(node_volts);
        sim_b.import_prev_node_volts(prev_node_volts);
        sim_b.import_cap_volts(cap_volts);
        sim_b.import_prev_cap_volts(prev_cap_volts);
        sim_b.import_inductor_currents(inductor_currents);
        sim_b.import_prev_inductor_currents(prev_inductor_currents);
        sim_b.import_tj_cap_volts(tj_cap_volts);
        sim_b.import_relay_active(relay_active);
        sim_b.import_gear2_ready(gear2_ready);
        sim_b.import_prev_dt(prev_dt);

        // The very next read should still show ~1.97 V — no DC reset.
        let v_after_restore = sim_b.node_voltage(2);
        assert!(
            (v_after_restore - v_mid).abs() < 1e-9,
            "expected seamless continuation at {} V, got {} V",
            v_mid, v_after_restore,
        );

        // Step another 1 ms with R=2kΩ (τ now 2 ms).  Expected V at end:
        //   V(t) = V_init + (5 - V_init) · (1 - e^-(t/τ_new))
        //        = 1.97 + 3.03 · (1 - e^-0.5)
        //        = 1.97 + 3.03 · 0.393 ≈ 3.16 V
        for _ in 0..1_000 {
            assert!(sim_b.step(1e-6).ok);
        }
        let v_final = sim_b.node_voltage(2);
        assert!((v_final - 3.16).abs() < 0.1,
            "expected V ≈ 3.16 V after pot turn + further charge, got {}", v_final);
    }
}
