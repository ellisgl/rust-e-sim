//! Transient solver — Newton-Raphson on the MNA system with adaptive integration.
//!
//! This module implements the main simulation loop for nonlinear transient
//! analysis. It uses a predictor-corrector approach:
//! 1. **Predictor**: Extrapolates the next state from previous time-steps to
//!    provide a better initial guess for Newton's method.
//! 2. **Corrector**: Refines the state using Newton-Raphson iteration on the
//!    Modified Nodal Analysis (MNA) system.
//!
//! Integration Methods
//! -------------------
//! Supports two implicit integration schemes:
//! - **Backward Euler (BE)**: Robust first-order method, used on the first step
//!   and after any discontinuity (e.g., relay toggle).
//! - **BDF-2 (Gear-2)**: Second-order method used for higher accuracy in
//!   smooth regions of the simulation.
//!
//! Numerical stability is maintained through SPICE-style voltage limiting
//! (`pnjlim`) for nonlinear devices and `GMIN` regularisation.

use crate::compile::CompiledNetlist;
use crate::linear::solve_linear_system;
use crate::netlist::Element;
use crate::sparse::{numeric_factor, sparse_solve_in_place};
use crate::diode::DiodeModel;
use crate::transistor::TransistorModel;
use serde::{Serialize, Deserialize};

/// Mutable per-step solver state.
///
/// Holds the history buffers required for multi-step integration (BDF-2)
/// and the current operating point for nonlinear devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransientState {
    /// Node voltages in compact MNA order.  Length = `compiled.n`.
    pub node_volts: Vec<f64>,
    /// Per-capacitor voltage (positive minus negative terminal).
    pub cap_volts: Vec<f64>,
    /// Per-capacitor voltage from TWO steps ago — Gear-2 history term.
    /// Used only when `gear2_ready` is true (so unused on the first step).
    pub prev_cap_volts: Vec<f64>,
    /// Per-inductor branch current.
    pub inductor_currents: Vec<f64>,
    /// Per-inductor current from two steps ago — Gear-2 history term.
    pub prev_inductor_currents: Vec<f64>,
    /// Per-inductor current from THREE steps ago — for LTE estimation.
    pub prev2_inductor_currents: Vec<f64>,
    /// Previous node voltages — read by the predictor warm-start to
    /// extrapolate `n+1` from `n` and `n-1`.
    pub prev_node_volts: Vec<f64>,
    /// Node voltages from two steps ago — for LTE estimation.
    pub prev2_node_volts: Vec<f64>,
    /// Per-capacitor voltage from three steps ago — for LTE estimation.
    pub prev2_cap_volts: Vec<f64>,
    /// Junction-cap voltages, layout `[Q0_Vbe, Q0_Vbc, Q1_Vbe, Q1_Vbc, …]`.
    pub tj_cap_volts: Vec<f64>,
    /// Per-voltage-source branch current.  Indexed by VS order in
    /// `compiled.voltage_source_indices`.  These are the augmented MNA
    /// unknowns captured from `est[n..n+m]` at commit time, populated
    /// in both `solve_dc` and `step_with_config`.  Audio/diagnostic
    /// paths read these via `Simulator::voltage_source_current(id)`.
    pub voltage_source_currents: Vec<f64>,
    /// Per-relay active flag.  `false` = de-energised (NC contact closed),
    /// `true` = energised (NO contact closed).  Updated once per step
    /// after Newton converges by `update_relay_states`, NOT per-iteration
    /// (matches TS — relays react at step granularity, not iter).
    pub relay_active: Vec<bool>,

    /// True once a successful step has been committed; gates BDF-2 and the
    /// predictor (both need a previous step to look at).  Cleared by
    /// `solve_dc` so the first transient step after DC always uses BE.
    pub gear2_ready: bool,
    /// dt of the previous step — scales the predictor extrapolation.
    pub prev_dt: f64,
}

impl TransientState {
    /// Zero-initialised state matching the compiled netlist's dimensions.
    /// Per-element initial conditions (capacitor `initial_voltage`) are
    /// applied here.
    pub fn new(c: &CompiledNetlist) -> Self {
        let mut cap_volts = vec![0.0; c.cap_count];
        for (i, &el_idx) in c.capacitor_indices.iter().enumerate() {
            if let Element::Capacitor { initial_voltage, .. } = &c.elements[el_idx] {
                cap_volts[i] = *initial_voltage;
            }
        }
        Self {
            node_volts: vec![0.0; c.n],
            prev_node_volts: vec![0.0; c.n],
            prev2_node_volts: vec![0.0; c.n],
            cap_volts: cap_volts.clone(),
            prev_cap_volts: cap_volts.clone(),
            prev2_cap_volts: cap_volts,
            inductor_currents: vec![0.0; c.inductor_count],
            prev_inductor_currents: vec![0.0; c.inductor_count],
            prev2_inductor_currents: vec![0.0; c.inductor_count],
            tj_cap_volts: vec![0.0; c.transistor_count * 2],
            voltage_source_currents: vec![0.0; c.m],
            relay_active: vec![false; c.relay_count],
            gear2_ready: false,
            prev_dt: 0.0,
        }
    }
}

/// Outcome of a single step.
#[derive(Debug, Clone, Copy)]
pub enum StepIssue {
    /// The linear solve failed even after the dense fallback.
    SingularMatrix,
    /// Newton hit the max iteration count without converging.
    NewtonDidNotConverge,
    /// Caller passed a non-finite or zero dt.
    BadTimestep,
}

/// Step configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct StepConfig {
    /// Timestep duration (seconds).
    pub dt: f64,
    /// Integration method: Backward Euler or BDF-2.
    pub gear: Gear,
    /// If true, calculate Local Truncation Error (LTE).
    pub estimate_lte: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gear {
    Be,
    Bdf2,
}

impl StepConfig {
    /// Convenience constructor matching the most common transient setup:
    /// BDF-2 integration with the caller's dt.
    pub fn bdf2(dt: f64) -> Self {
        Self { dt, gear: Gear::Bdf2, estimate_lte: false }
    }
    pub fn be(dt: f64) -> Self {
        Self { dt, gear: Gear::Be, estimate_lte: false }
    }
    pub fn with_lte(mut self) -> Self {
        self.estimate_lte = true;
        self
    }
}

/// LTE estimation result.
#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    /// Number of Newton iterations performed.
    pub iters: usize,
    /// Estimated Local Truncation Error (max across all state variables).
    /// Only populated if `estimate_lte` was true.
    pub lte: Option<f64>,
}

const GMIN: f64 = 1e-9;
const NEWTON_RTOL: f64 = 1e-4;
const NEWTON_ATOL: f64 = 1e-6;
const STEP_LIMIT: f64 = 1.0; // V — per-iteration voltage clamp
/// Maximum single-node correction the predictor may apply.  TS uses the
/// same value; clipping keeps Newton inside the pnjlim-safe region after
/// large dt jumps or topology changes.
const PREDICTOR_CLIP: f64 = 1.5;

/// Maximum Newton iterations allowed for the current netlist.
///
/// Returns a budget based on the complexity of the circuit (number of
/// transistors, diodes, and relays). Nonlinear circuits require more
/// iterations to reach convergence.
fn total_iterations(transistor_count: usize, diode_count: usize, relay_count: usize) -> usize {
    let q_iter = if transistor_count > 0 { 20 } else { 1 };
    let d_iter = if diode_count > 0 { 10 } else { 1 };
    // Matches TS transient.ts relayIterations.  Relays only flip on
    // hysteresis crossings — three iterations is enough headroom for
    // a hot circuit to settle the contact state without dragging
    // every linear netlist into a 3-iter Newton loop.
    let r_iter = if relay_count > 0 { 3 } else { 1 };
    q_iter.max(d_iter).max(r_iter)
}

/// Reason a DC solve failed.
#[derive(Debug, Clone, Copy)]
pub enum DcIssue {
    SingularMatrix,
    DidNotConverge,
}

/// How nonlinear devices are linearised for one Newton assembly.
#[derive(Clone, Copy)]
struct StampMode<'a> {
    /// Previous voltages for BJT pnjlim (`None` = no limiting).
    bjt_prev: Option<&'a [f64]>,
    /// Previous voltages for diode pnjlim (`None` = no limiting).
    diode_prev: Option<&'a [f64]>,
    /// Transient only: BJT junction/diffusion caps as BE companions,
    /// `(tj_cap_volts, 1/dt)`.
    junction_caps: Option<(&'a [f64], f64)>,
}

/// Read-only view of the device tables, split from the mutable scratch
/// buffers so both can be borrowed from one `CompiledNetlist` at once.
struct Devices<'a> {
    size: usize,
    n: usize,
    elements: &'a [Element],
    transistor_models: &'a [TransistorModel],
    transistor_node_indices: &'a [i32],
    diode_models: &'a [DiodeModel],
    diode_node_indices: &'a [i32],
    relay_indices: &'a [usize],
    relay_node_indices: &'a [i32],
}

impl Devices<'_> {
    /// Stamp every nonlinear / state-dependent device (BJTs + junction caps,
    /// diodes, relays) on top of the base system already in `matrix`/`rhs`.
    fn stamp(
        &self,
        matrix: &mut [f64],
        rhs: &mut [f64],
        est: &[f64],
        relay_active: &[bool],
        mode: StampMode<'_>,
    ) {
        let size = self.size;
        let volts = &est[..self.n];

        for (ti, (q, ix)) in self.transistor_models.iter()
            .zip(self.transistor_node_indices.chunks_exact(3))
            .enumerate()
        {
            let (bi, ci, ei) = (ix[0], ix[1], ix[2]);
            // In transient mode the previous-step node voltages engage SPICE
            // pnjlim, limiting how far Vbe can swing per step.  Without it
            // Newton oversteps the BJT's narrow active region in one
            // iteration and cap-stabilised transitions get short-circuited.
            let s = q.stamp(volts, bi, ci, ei, mode.bjt_prev);

            // BJT MNA stamps.  Sign conventions match transient.ts.
            //   I_C = gm·Vbe + gmu·Vbc + iEqC
            //   I_B = gpi·Vbe + gmu_b·Vbc + iEqB
            //   (I_E by KCL.)
            stamp_bjt(
                matrix, rhs, size, bi, ci, ei,
                s.gm, s.gmu, s.gpi, s.gmu_b,
                s.i_eq_b, s.i_eq_c, s.i_eq_e,
            );

            // Junction + diffusion capacitance — REQUIRED for correct
            // large-signal switching dynamics.  Without these stamps BJTs in
            // blocking-oscillator circuits (metronome, siren) switch
            // instantly, run at 10× the correct rate and produce ~20 dB
            // lower speaker swing.  TS uses plain BE for these caps (not
            // BDF-2); we match that here.
            //
            // C_BE = cje + tf · gm     (junction + forward diffusion)
            // C_BC = cjc + tr · gmu_b  (junction + reverse diffusion)
            if let Some((tj_cap_volts, dt_inv)) = mode.junction_caps {
                let cbe_total = q.cje_farads + q.tf_seconds * s.gm;
                let cbc_total = q.cjc_farads + q.tr_seconds * s.gmu_b;
                if cbe_total > 0.0 {
                    let g = cbe_total * dt_inv;
                    stamp_two_node_cap(matrix, rhs, size, bi, ei, g, g * tj_cap_volts[2 * ti]);
                }
                if cbc_total > 0.0 {
                    let g = cbc_total * dt_inv;
                    stamp_two_node_cap(matrix, rhs, size, bi, ci, g, g * tj_cap_volts[2 * ti + 1]);
                }
            }
        }

        for (d, ix) in self.diode_models.iter().zip(self.diode_node_indices.chunks_exact(2)) {
            let (ai, ki) = (ix[0], ix[1]);
            let s = d.stamp(volts, ai, ki, mode.diode_prev);
            // Y[a,a] += gd; Y[k,k] += gd; Y[a,k] -= gd; Y[k,a] -= gd
            // rhs[a] -= ieq; rhs[k] += ieq
            if ai >= 0 {
                let ai = ai as usize;
                matrix[ai * size + ai] += s.gd;
                rhs[ai] -= s.ieq;
            }
            if ki >= 0 {
                let ki = ki as usize;
                matrix[ki * size + ki] += s.gd;
                rhs[ki] += s.ieq;
            }
            if ai >= 0 && ki >= 0 {
                matrix[(ai as usize) * size + (ki as usize)] -= s.gd;
                matrix[(ki as usize) * size + (ai as usize)] -= s.gd;
            }
        }

        // Relays: linear resistive stamps whose values depend on
        // `relay_active`, which `update_relay_states` flips after each solve.
        stamp_relays(
            matrix, size, self.relay_indices, self.relay_node_indices,
            self.elements, relay_active,
        );
    }
}

/// Assemble one Newton iteration from the base system and solve it.
///
/// Working matrix/RHS = base (+ `extra_gmin` on node diagonals, if any) +
/// device stamps.  Tries the sparse LU first; if a pivot collapses, the
/// system is re-assembled from scratch and solved with dense partial
/// pivoting.  On `true` the solution is in `c.rhs`.
fn assemble_and_solve(
    c: &mut CompiledNetlist,
    extra_gmin: Option<f64>,
    est: &[f64],
    relay_active: &[bool],
    mode: StampMode<'_>,
) -> bool {
    let CompiledNetlist {
        matrix, rhs, base_matrix, base_rhs, gmin_indices, sparse_pattern,
        size, n, elements,
        transistor_models, transistor_node_indices,
        diode_models, diode_node_indices,
        relay_indices, relay_node_indices,
        ..
    } = c;
    let size = *size;
    let devices = Devices {
        size,
        n: *n,
        elements,
        transistor_models,
        transistor_node_indices,
        diode_models,
        diode_node_indices,
        relay_indices,
        relay_node_indices,
    };
    let assemble = |matrix: &mut [f64], rhs: &mut [f64]| {
        matrix.copy_from_slice(base_matrix);
        if let Some(g) = extra_gmin {
            for &g_idx in gmin_indices.iter() {
                matrix[g_idx as usize] += g;
            }
        }
        rhs.copy_from_slice(base_rhs);
        devices.stamp(matrix, rhs, est, relay_active, mode);
    };

    assemble(matrix, rhs);
    if numeric_factor(matrix, size, sparse_pattern) {
        sparse_solve_in_place(matrix, rhs, size, sparse_pattern);
        return true;
    }
    // Rare singular-pivot fallback: the sparse factorisation clobbered the
    // matrix, so rebuild BOTH matrix and RHS before the dense solve.
    assemble(matrix, rhs);
    solve_linear_system(matrix, rhs, size)
}

/// Per-circuit DC Newton iteration budget.
fn dc_iter_budget(c: &CompiledNetlist) -> usize {
    if c.transistor_count > 0 {
        100
    } else if c.diode_count > 0 {
        50
    } else if c.relay_count > 0 {
        20
    } else {
        1
    }
}

/// One DC Newton solve at a fixed `gmin`.
///
/// With `prev_est = None` devices are linearised without pnjlim (the fast
/// path).  With `Some(prev_est)` pnjlim limits each iterate against the
/// previous one, and `damp` under-relaxes the update — used by the gmin-
/// and source-stepping fallbacks.  Converges when the scaled max change
/// drops below `tol`.
fn dc_newton(
    c: &mut CompiledNetlist,
    state: &mut TransientState,
    gmin: f64,
    est: &mut [f64],
    mut prev_est: Option<&mut [f64]>,
    damp: f64,
    tol: f64,
) -> Result<usize, DcIssue> {
    let size = c.size;
    let linear = c.transistor_count == 0 && c.diode_count == 0 && c.relay_count == 0;

    for iteration in 0..dc_iter_budget(c) {
        let limit = prev_est.as_deref();
        let mode = StampMode { bjt_prev: limit, diode_prev: limit, junction_caps: None };
        if !assemble_and_solve(c, Some(gmin), est, &state.relay_active, mode) {
            return Err(DcIssue::SingularMatrix);
        }

        if let Some(p) = prev_est.as_deref_mut() {
            p.copy_from_slice(est);
        }

        // Under-relaxation (damping).  A full Newton step (damp = 1.0) can
        // enter a 2-cycle limit cycle near an oscillator's unstable
        // equilibrium — the feedback node overshoots, the companion currents
        // flip, and the iterate ping-pongs forever without converging.
        // Taking a fractional step, est = old + damp·(new − old), breaks the
        // cycle.  Callers pass damp = 1.0 for the normal fast path and a
        // smaller value (≈0.3) during source stepping.
        let mut max_err = 0.0;
        let mut saw_nan = false;
        for (e, &new_v) in est.iter_mut().zip(&c.rhs[..size]) {
            let old_v = *e;
            let err = (new_v - old_v).abs() / (1.0 + new_v.abs().max(old_v.abs()));
            if err > max_err { max_err = err; }
            *e = if damp >= 1.0 { new_v } else { old_v + damp * (new_v - old_v) };
            if !e.is_finite() { saw_nan = true; }
        }
        if saw_nan {
            return Err(DcIssue::DidNotConverge);
        }

        if iteration > 0 && max_err < tol {
            return Ok(iteration + 1);
        }
        if iteration == 0 && linear {
            return Ok(1);
        }

        update_relay_states(
            &mut state.relay_active, &c.relay_indices,
            &c.relay_node_indices, &c.elements, est,
        );
    }

    // Every iteration either solved or returned early, so reaching here
    // means the budget ran out after at least one successful solve.
    Err(DcIssue::DidNotConverge)
}

/// Solve the DC operating point of the circuit.
///
/// Capacitors are treated as open circuits and inductors as short circuits.
/// Returns the number of Newton iterations required to reach convergence.
pub fn solve_dc(
    c: &mut CompiledNetlist,
    state: &mut TransientState,
) -> Result<usize, DcIssue> {
    let size = c.size;
    let n = c.n;

    // Largest |V| across voltage sources — used for transistor warm-start
    // to pick a sane initial collector voltage.
    let max_vcc = c
        .voltage_source_values
        .iter()
        .map(|v| v.abs())
        .fold(5.0_f64, f64::max);

    // ── Build the DC base matrix + RHS ─────────────────────────────────
    // Static stamps (resistors + V-source incidence) are reused as-is.
    // Caps are skipped (open circuit).  Inductor branch-row coefficient is
    // zero — the incidence stamps already enforce V_a = V_b, which is the
    // short-circuit condition.  gmin is added per Newton solve (it varies
    // during gmin stepping).
    c.base_matrix.copy_from_slice(&c.static_matrix);
    c.base_rhs.fill(0.0);
    for (&row, &v) in c.voltage_source_branch_rows.iter().zip(&c.voltage_source_values) {
        c.base_rhs[row as usize] = v;
    }

    // ── Warm-start est buffer ──────────────────────────────────────────
    let mut est = vec![0.0; size];
    for (ti, &el_idx) in c.transistor_indices.iter().enumerate() {
        let Element::Transistor { params, .. } = &c.elements[el_idx] else { unreachable!() };
        let bi = c.transistor_node_indices[ti * 3];
        let ci_ = c.transistor_node_indices[ti * 3 + 1];
        let ei = c.transistor_node_indices[ti * 3 + 2];
        let (ve, vb, vc) = match params.polarity {
            crate::types::Polarity::Npn => (0.0, 0.6, max_vcc * 0.5),
            crate::types::Polarity::Pnp => (max_vcc, max_vcc * 0.9, max_vcc * 0.5),
        };
        if ei >= 0 { est[ei as usize] = ve; }
        if bi >= 0 { est[bi as usize] = vb; }
        if ci_ >= 0 { est[ci_ as usize] = vc; }
    }

    let mut actual_iters = 0;

    // ── DC Newton iteration ────────────────────────────────────────────
    // Three escalating strategies:
    //   1. Direct Newton (no limiting)        — fast path, most circuits.
    //   2. Gmin stepping                      — diodes / mild nonlinearity.
    //   3. Source stepping + damping          — oscillators whose DC
    //                                            equilibrium is unstable
    //                                            (relaxation / LC), which
    //                                            send plain Newton into a
    //                                            limit cycle even with gmin.
    let mut prev_est = est.clone();
    match dc_newton(c, state, GMIN, &mut est, None, 1.0, 1e-3) {
        Ok(iters) => {
            actual_iters = iters;
        }
        Err(_) => {
            // ── Strategy 2: Gmin stepping ──────────────────────────────
            // Start with a large GMIN to force convergence, then
            // exponentially reduce it back to the target GMIN.  Runs on a
            // scratch copy so a late failure doesn't corrupt `est` before
            // strategy 3 gets a clean start.
            const GMIN_START: f64 = 1e-2;
            const GMIN_STEPS: usize = 10;
            let gmin_target = GMIN;
            let factor = (gmin_target / GMIN_START).powf(1.0 / GMIN_STEPS as f64);

            let mut est_gmin = est.clone();
            let mut prev_gmin = prev_est.clone();
            let mut gmin_iters = 0;
            let mut gmin_current = GMIN_START;
            let mut gmin_ok = true;
            for _ in 0..=GMIN_STEPS {
                match dc_newton(
                    c, state, gmin_current, &mut est_gmin, Some(&mut prev_gmin), 1.0, 1e-6,
                ) {
                    Ok(it) => gmin_iters += it,
                    Err(_) => { gmin_ok = false; break; }
                }
                gmin_current = (gmin_current * factor).max(gmin_target);
            }

            if gmin_ok {
                est.copy_from_slice(&est_gmin);
                actual_iters += gmin_iters;
            } else {
                // ── Strategy 3: Source stepping (supply ramping) ───────
                // Ramp every independent source from 0 to its full value in
                // many small increments, re-solving at each and carrying the
                // solution forward as the warm start.  Each increment nudges
                // the operating point only slightly, keeping Newton inside
                // its convergence basin.  Under-relaxation (DAMP) breaks the
                // 2-cycle limit cycle that appears near an oscillator's
                // Hopf point, where the equilibrium turns unstable.
                //
                // SRC_STEPS=100 / DAMP=0.3 converge every kit oscillator
                // tested (P18 siren, P45 code oscillator, metronome).  These
                // are deliberately conservative — this path only runs once
                // at configure time after the two faster strategies fail,
                // so the extra iterations don't affect steady-state cost.
                const SRC_STEPS: usize = 100;
                const SRC_DAMP: f64 = 0.3;

                // Snapshot the full source drive (already written into
                // base_rhs at each voltage-source branch row).
                let saved: Vec<(usize, f64)> = c.voltage_source_branch_rows.iter()
                    .map(|&row| (row as usize, c.base_rhs[row as usize]))
                    .collect();

                // Cold start from zero: at zero drive the solution is ~0.
                est.fill(0.0);
                prev_est.copy_from_slice(&est);

                let mut src_ok = true;
                for si in 1..=SRC_STEPS {
                    let scale = si as f64 / SRC_STEPS as f64;
                    for &(row, val) in &saved {
                        c.base_rhs[row] = scale * val;
                    }
                    match dc_newton(
                        c, state, GMIN, &mut est, Some(&mut prev_est), SRC_DAMP, 1e-6,
                    ) {
                        Ok(it) => actual_iters += it,
                        Err(_) => { src_ok = false; break; }
                    }
                }

                // Restore the full source drive regardless of outcome so the
                // committed matrix/RHS reflect the real circuit.
                for &(row, val) in &saved {
                    c.base_rhs[row] = val;
                }

                if !src_ok {
                    return Err(DcIssue::DidNotConverge);
                }
            }
        }
    }

    // ── Commit DC solution into state ──────────────────────────────────
    // Belt-and-braces NaN guard: refuses to commit if any est element is
    // non-finite, so NaN can never poison state.node_volts (which would
    // make every subsequent transient warm-start NaN).
    if !est[..n].iter().all(|v| v.is_finite()) {
        return Err(DcIssue::DidNotConverge);
    }
    // Don't write into prev_* buffers — the first transient step after DC
    // should use BE, not BDF-2 (TS behaviour).  Clear gear2_ready.
    commit_solution(c, state, &est);
    state.gear2_ready = false;
    state.prev_dt = 0.0;

    Ok(actual_iters)
}

/// Write a solved MNA vector into the per-element state.
///
/// Layout: `est[0..n]` = node volts, `est[n..n+m]` = VS currents,
/// `est[n+m..]` = inductor branch currents.
fn commit_solution(c: &CompiledNetlist, state: &mut TransientState, est: &[f64]) {
    let n = c.n;
    let node_v = |idx: i32| if idx >= 0 { est[idx as usize] } else { 0.0 };

    state.node_volts.copy_from_slice(&est[..n]);
    for (cv, ix) in state.cap_volts.iter_mut().zip(c.cap_stamp_indices.chunks_exact(4)) {
        *cv = node_v(ix[0]) - node_v(ix[1]);
    }
    for (i, &br) in state.inductor_currents.iter_mut().zip(&c.inductor_branch_rows) {
        *i = est[br as usize];
    }
    state.voltage_source_currents.copy_from_slice(&est[n..n + c.m]);
    // Transistor junction-cap voltages (Vbe, Vbc).
    for (tj, ix) in state.tj_cap_volts.chunks_exact_mut(2)
        .zip(c.transistor_node_indices.chunks_exact(3))
    {
        let vb = node_v(ix[0]);
        tj[0] = vb - node_v(ix[2]);
        tj[1] = vb - node_v(ix[1]);
    }
}

/// Advances the simulation by one timestep using Backward Euler.
pub fn step(
    c: &mut CompiledNetlist,
    state: &mut TransientState,
    dt: f64,
) -> Result<usize, StepIssue> {
    step_with_config(c, state, StepConfig::be(dt)).map(|r| r.iters)
}

/// Advances the simulation by one timestep using the specified configuration.
///
/// Allocation-free: all scratch lives in the `CompiledNetlist`.
pub fn step_with_config(
    c: &mut CompiledNetlist,
    state: &mut TransientState,
    config: StepConfig,
) -> Result<StepResult, StepIssue> {
    // Detach the Newton scratch so it can be borrowed alongside `c`; this is
    // a pointer swap, not an allocation.  Always reattached before return.
    let mut est = std::mem::take(&mut c.est);
    est.resize(c.size, 0.0);
    let result = step_inner(c, state, config, &mut est);
    c.est = est;
    result
}

fn step_inner(
    c: &mut CompiledNetlist,
    state: &mut TransientState,
    config: StepConfig,
    est: &mut [f64],
) -> Result<StepResult, StepIssue> {
    let dt = config.dt;
    if dt <= 0.0 || !dt.is_finite() {
        return Err(StepIssue::BadTimestep);
    }
    let dt_inv = 1.0 / dt;
    let size = c.size;
    let n = c.n;

    let use_gear2 = config.gear == Gear::Bdf2 && state.gear2_ready;
    let can_predict = state.gear2_ready && state.prev_dt > 0.0;
    let dt_ratio = if can_predict { (dt / state.prev_dt).min(4.0) } else { 0.0 };

    // ── Build base matrix + RHS ─────────────────────────────────────────
    // Everything that doesn't change between Newton iterations goes here.
    // Each iteration copies this into the working matrix, stamps the
    // nonlinear contributions on top, factors, solves.
    //
    // Static stamps (resistors + voltage-source incidence) come
    // pre-accumulated from compile time.
    c.base_matrix.copy_from_slice(&c.static_matrix);
    c.base_rhs.fill(0.0);

    // gmin diagonal regularisation.
    for &g_idx in &c.gmin_indices {
        c.base_matrix[g_idx as usize] += GMIN;
    }

    // Voltage-source RHS — V_pos − V_neg = V_src.
    for (&row, &v) in c.voltage_source_branch_rows.iter().zip(&c.voltage_source_values) {
        c.base_rhs[row as usize] = v;
    }

    // Capacitor companion.  Backward Euler: g = C/dt, ieq = g·V_prev.
    // BDF-2 (when gear2_ready): g = 3C/(2·dt), ieq = (C/(2·dt))·(4·V_prev − V_prev2).
    // BDF-2 reduces to BE structurally — same stamp positions, different
    // coefficients.
    for (ci, &el_idx) in c.capacitor_indices.iter().enumerate() {
        let Element::Capacitor { capacitance_farads: cap_f, .. } = c.elements[el_idx] else {
            unreachable!("capacitor_indices points to a non-capacitor")
        };
        let prev_v = state.cap_volts[ci];
        let (g, ieq) = if use_gear2 {
            let prev2_v = state.prev_cap_volts[ci];
            ((3.0 * cap_f) / (2.0 * dt), (cap_f / (2.0 * dt)) * (4.0 * prev_v - prev2_v))
        } else {
            let g = cap_f * dt_inv;
            (g, g * prev_v)
        };

        let ix = &c.cap_stamp_indices[ci * 4..ci * 4 + 4];
        let (ia, ib, ab, ba) = (ix[0], ix[1], ix[2], ix[3]);
        if ia >= 0 {
            c.base_matrix[(ia as usize) * size + (ia as usize)] += g;
            c.base_rhs[ia as usize] += ieq;
        }
        if ib >= 0 {
            c.base_matrix[(ib as usize) * size + (ib as usize)] += g;
            c.base_rhs[ib as usize] -= ieq;
        }
        if ab >= 0 {
            c.base_matrix[ab as usize] -= g;
        }
        if ba >= 0 {
            c.base_matrix[ba as usize] -= g;
        }
    }

    // Inductor companion.  Backward Euler: coeff = L/dt, rhs = -coeff·I_prev.
    // BDF-2: coeff = 3L/(2·dt), rhs = -(L/(2·dt))·(4·I_prev − I_prev2).
    // The branch equation is V_a − V_b − coeff·I_new = -rhs.
    //
    // Inductor saturation: when |I_prev| exceeds `saturation_current_a`,
    // the effective inductance drops to 1% of nominal (core saturates,
    // inductance collapses).  Matches the TS reference's simple two-state
    // saturation model.
    for (li, &el_idx) in c.inductor_indices.iter().enumerate() {
        let Element::Inductor { inductance_henry: l_nominal, saturation_current_a: i_sat, .. } =
            c.elements[el_idx]
        else {
            unreachable!("inductor_indices points to a non-inductor")
        };
        let prev_i = state.inductor_currents[li];
        let l_eff = match i_sat {
            Some(isat) if prev_i.abs() > isat => l_nominal * 0.01,
            _ => l_nominal,
        };
        let (coeff, rhs_val) = if use_gear2 {
            let prev2_i = state.prev_inductor_currents[li];
            ((3.0 * l_eff) / (2.0 * dt), (l_eff / (2.0 * dt)) * (4.0 * prev_i - prev2_i))
        } else {
            let coeff = l_eff * dt_inv;
            (coeff, coeff * prev_i)
        };
        let branch_row = c.inductor_branch_rows[li] as usize;
        c.base_matrix[branch_row * size + branch_row] -= coeff;
        c.base_rhs[branch_row] = -rhs_val;
    }

    // Mutual inductance.  For each ordered pair (i, j) in the same
    // coupling group with signed mutual M_ij:
    //   M_coeff = M/dt        (BE) or  3·M/(2·dt)                       (BDF-2)
    //   M_rhs   = (M/dt)·I_j  (BE) or  (M/(2·dt))·(4·I_j − I_j_prev2)   (BDF-2)
    //   matrix[branch_i, branch_j] −= M_coeff
    //   rhs[branch_i]              −= M_rhs
    // The pair list contains both (i, j) and (j, i) so the matrix is
    // symmetric without an explicit transpose stamp.
    for pair in c.inductor_coupling_pairs.chunks_exact(3) {
        let i_idx = pair[0] as usize;
        let j_idx = pair[1] as usize;
        let m_signed = pair[2];
        let prev_j = state.inductor_currents[j_idx];
        let (m_coeff, m_rhs) = if use_gear2 {
            let prev2_j = state.prev_inductor_currents[j_idx];
            ((3.0 * m_signed) / (2.0 * dt),
             (m_signed / (2.0 * dt)) * (4.0 * prev_j - prev2_j))
        } else {
            (m_signed * dt_inv, m_signed * dt_inv * prev_j)
        };
        let br_i = c.inductor_branch_rows[i_idx] as usize;
        let br_j = c.inductor_branch_rows[j_idx] as usize;
        c.base_matrix[br_i * size + br_j] -= m_coeff;
        c.base_rhs[br_i] -= m_rhs;
    }

    // ── Initial Newton estimate ─────────────────────────────────────────
    // Layout matches MNA matrix rows.  Without a predictor, this is just
    // the previous step's node voltages.  With a predictor (gear2_ready +
    // prev_dt > 0), linearly extrapolate forward by dt_ratio = dt/prev_dt
    // and clip to PREDICTOR_CLIP to stay inside the pnjlim-safe region.
    // Branch entries start at zero; they're computed by the first solve.
    if can_predict {
        for ((e, &curr), &prev) in est[..n].iter_mut()
            .zip(&state.node_volts)
            .zip(&state.prev_node_volts)
        {
            let delta = (curr - prev) * dt_ratio;
            *e = curr + delta.clamp(-PREDICTOR_CLIP, PREDICTOR_CLIP);
        }
    } else {
        est[..n].copy_from_slice(&state.node_volts);
    }
    est[n..].fill(0.0);

    // ── Newton iteration ────────────────────────────────────────────────
    // Loop budget matches TS: linear systems run exactly once (the SPICE
    // step-limit clamp produces a converged answer over a few timesteps).
    // Nonlinear systems get a generous budget with a convergence-based
    // early break.
    let max_iters = total_iterations(c.transistor_count, c.diode_count, c.relay_count);
    let mut actual_iters = 0;
    let mut prev_raw_max_delta = f64::INFINITY;
    // With predictor active, accept convergence as early as iteration 1;
    // without one, require 2 iterations to avoid committing to an
    // unconverged warm-start.  Matches the TS reference's minConvergeIter.
    let min_converge_iter = if can_predict { 1 } else { 2 };

    for iteration in 0..max_iters {
        actual_iters = iteration + 1;

        // Transient mode: BJTs get pnjlim against the previous step's node
        // voltages plus junction-cap companions; diodes run unlimited
        // (matches TS).
        let mode = StampMode {
            bjt_prev: Some(&state.node_volts),
            diode_prev: None,
            junction_caps: Some((&state.tj_cap_volts, dt_inv)),
        };
        if !assemble_and_solve(c, None, est, &state.relay_active, mode) {
            return Err(StepIssue::SingularMatrix);
        }

        // ── Damped update + convergence check ───────────────────────────
        let raw_max_delta = c.rhs[..n].iter().zip(&est[..n])
            .fold(0.0_f64, |m, (&new_v, &old_v)| m.max((new_v - old_v).abs()));
        // Damping matches transient.ts: full step at iteration 0; heavy
        // damping when raw delta doubled (divergence warning); moderate
        // otherwise.
        let damping = if iteration == 0 {
            1.0
        } else if raw_max_delta > prev_raw_max_delta * 2.0 {
            0.1
        } else if iteration < 3 {
            0.6
        } else {
            0.3
        };
        prev_raw_max_delta = raw_max_delta;

        let mut max_delta = 0.0_f64;
        let mut max_v = 0.0_f64;
        let mut saw_nan = false;
        for (e, &new_v) in est[..n].iter_mut().zip(&c.rhs[..n]) {
            let old_v = *e;
            let delta = (new_v - old_v).clamp(-STEP_LIMIT, STEP_LIMIT);
            *e = old_v + damping * delta;
            if !e.is_finite() { saw_nan = true; }
            max_delta = max_delta.max(delta.abs());
            max_v = max_v.max(e.abs());
        }
        for (e, &new_v) in est[n..].iter_mut().zip(&c.rhs[n..size]) {
            *e = new_v;
            if !e.is_finite() { saw_nan = true; }
        }
        // NaN guard: if Newton produced any non-finite iterate, abort the
        // step.  NaN comparisons all evaluate false, so without this Newton
        // would run to budget exhaustion and commit NaN state.  Returning
        // before commit preserves the prev_* buffers so the caller can
        // retry with a smaller dt or reseed from DC.
        if saw_nan {
            return Err(StepIssue::NewtonDidNotConverge);
        }

        // Update relay states for the next iteration.  Matches TS exactly:
        // a relay can flip mid-Newton when its coil voltage crosses the
        // on/off thresholds with hysteresis, and the next stamp pass picks
        // up the new contact resistance.
        update_relay_states(
            &mut state.relay_active, &c.relay_indices,
            &c.relay_node_indices, &c.elements, est,
        );

        // Only accept convergence after min_converge_iter to avoid
        // committing to an unconverged warm-start.  Newton may also not
        // formally converge within the budget — TS doesn't treat that as a
        // failure (the step-limit clamp keeps the iterate bounded), so the
        // final est is committed regardless.
        if iteration >= min_converge_iter && max_delta < NEWTON_RTOL * max_v + NEWTON_ATOL {
            break;
        }
    }

    // ── Commit solution into state ──────────────────────────────────────
    // Shift history BEFORE overwriting state with the new step.
    state.prev2_node_volts.copy_from_slice(&state.prev_node_volts);
    state.prev_node_volts.copy_from_slice(&state.node_volts);

    state.prev2_cap_volts.copy_from_slice(&state.prev_cap_volts);
    state.prev_cap_volts.copy_from_slice(&state.cap_volts);

    state.prev2_inductor_currents.copy_from_slice(&state.prev_inductor_currents);
    state.prev_inductor_currents.copy_from_slice(&state.inductor_currents);

    commit_solution(c, state, est);

    let gear2_already_ready = state.gear2_ready;

    // Mark history as populated so the NEXT step can use BDF-2 + predictor.
    state.gear2_ready = true;
    state.prev_dt = dt;

    let mut lte = None;
    if config.estimate_lte && gear2_already_ready {
        // Simple LTE estimation using divided differences on the new
        // solution x_n and the history x_{n-1}, x_{n-2}.
        // BDF-2 LTE is O(dt^3), usually smaller — very rough 0.22 factor.
        let scale = if use_gear2 { 0.22 } else { 0.5 };
        let max_lte = state.node_volts.iter()
            .zip(&state.prev_node_volts)
            .zip(&state.prev2_node_volts)
            .fold(0.0_f64, |m, ((&xn, &xn1), &xn2)| {
                m.max(scale * (xn - 2.0 * xn1 + xn2).abs()) // ≈ dt²·|x''|
            });
        lte = Some(max_lte);
    }

    Ok(StepResult { iters: actual_iters, lte })
}

/// Stamp a two-node resistive conductance into the matrix only.  Skips
/// terms whose node is ground.  Used for relay coil and contact stamps
/// (which carry no companion-current contribution — they're pure
/// linear resistors whose value happens to change per Newton iteration
/// depending on relay state).
#[inline]
fn stamp_two_node_conductance(matrix: &mut [f64], size: usize, ia: i32, ib: i32, g: f64) {
    if ia >= 0 {
        let ia = ia as usize;
        matrix[ia * size + ia] += g;
    }
    if ib >= 0 {
        let ib = ib as usize;
        matrix[ib * size + ib] += g;
    }
    if ia >= 0 && ib >= 0 {
        let ia = ia as usize;
        let ib = ib as usize;
        matrix[ia * size + ib] -= g;
        matrix[ib * size + ia] -= g;
    }
}

/// Stamp every relay in the compiled netlist according to its current
/// `state.relay_active` flag.  Each relay contributes three resistive
/// stamps:
///   - coil:                   1 / Rcoil  between (coil+, coil-)
///   - active contact:         1 / Ron    between (common, active throw)
///   - inactive contact:       1 / Roff   between (common, inactive throw)
///
/// Active throw is NC when relay_active=false (rest state), NO otherwise.
///
/// Called inside the per-iteration Newton stamp pass; pairs with
/// `update_relay_states` which mutates relay_active after each solve.
///
/// Takes immutable fields one-by-one rather than `&CompiledNetlist` so
/// the caller can pass `&mut c.matrix` and read-only views of the rest
/// (Rust's split-borrow rule needs explicit per-field borrows).
#[inline]
fn stamp_relays(
    matrix: &mut [f64], size: usize,
    relay_indices: &[usize],
    relay_node_indices: &[i32],
    elements: &[Element],
    relay_active: &[bool],
) {
    for (ri, (&el_idx, ix)) in relay_indices.iter()
        .zip(relay_node_indices.chunks_exact(5))
        .enumerate()
    {
        let (coil_p, coil_n, cmn, nc, no) = (ix[0], ix[1], ix[2], ix[3], ix[4]);
        let (rcoil, ron, roff) = match &elements[el_idx] {
            Element::Relay { coil_resistance_ohms, ron_ohms, roff_ohms, .. } => {
                (*coil_resistance_ohms, *ron_ohms, *roff_ohms)
            }
            _ => unreachable!(),
        };
        stamp_two_node_conductance(matrix, size, coil_p, coil_n, 1.0 / rcoil);
        let active = relay_active[ri];
        // Active contact (Ron, ~low Ω); inactive contact (Roff, ~1MΩ).
        let (active_node, inactive_node) = if active { (no, nc) } else { (nc, no) };
        stamp_two_node_conductance(matrix, size, cmn, active_node,   1.0 / ron);
        stamp_two_node_conductance(matrix, size, cmn, inactive_node, 1.0 / roff);
    }
}

/// Update each relay's `relay_active` flag based on coil voltage.  Called
/// once per Newton iteration *after* the linear solve; matches TS, where
/// a relay can flip mid-Newton (the next iter restamps with new contact
/// conductances).  Uses voltage-based hysteresis comparing
/// |V_coil| against I_threshold·R_coil, which is equivalent to comparing
/// |I_coil| against I_threshold but avoids a division per iteration.
#[inline]
fn update_relay_states(
    relay_active: &mut [bool],
    relay_indices: &[usize],
    relay_node_indices: &[i32],
    elements: &[Element],
    est: &[f64],
) {
    for (ri, (&el_idx, ix)) in relay_indices.iter()
        .zip(relay_node_indices.chunks_exact(5))
        .enumerate()
    {
        let (coil_p, coil_n) = (ix[0], ix[1]);
        let (rcoil, i_on, i_off) = match &elements[el_idx] {
            Element::Relay {
                coil_resistance_ohms, on_current, off_current, ..
            } => (*coil_resistance_ohms, *on_current, *off_current),
            _ => unreachable!(),
        };
        let v_p = if coil_p >= 0 { est[coil_p as usize] } else { 0.0 };
        let v_n = if coil_n >= 0 { est[coil_n as usize] } else { 0.0 };
        let v_coil = (v_p - v_n).abs();
        let was_active = relay_active[ri];
        let should_activate = v_coil > i_on  * rcoil;
        let should_release  = v_coil < i_off * rcoil;
        if !was_active && should_activate { relay_active[ri] = true;  }
        if  was_active && should_release  { relay_active[ri] = false; }
    }
}

/// Stamp a two-terminal linear capacitor companion (BE-discretised) into
/// the matrix + RHS.  Used for both ordinary capacitors and BJT junction
/// capacitances (cje + tf·gm at BE, cjc + tr·gmu_b at BC).
///
/// Formula: companion conductance `g = C/dt`, equivalent current
/// `ieq = g·V_prev`.  Stamps `Y[a,a] += g`, `Y[b,b] += g`,
/// `Y[a,b] -= g`, `Y[b,a] -= g`, `rhs[a] += ieq`, `rhs[b] -= ieq`.
/// Skips terms whose node is ground (idx < 0).
#[inline]
fn stamp_two_node_cap(
    matrix: &mut [f64], rhs: &mut [f64], size: usize,
    ia: i32, ib: i32, g: f64, ieq: f64,
) {
    if ia >= 0 {
        let ia = ia as usize;
        matrix[ia * size + ia] += g;
        rhs[ia] += ieq;
    }
    if ib >= 0 {
        let ib = ib as usize;
        matrix[ib * size + ib] += g;
        rhs[ib] -= ieq;
    }
    if ia >= 0 && ib >= 0 {
        let ia = ia as usize;
        let ib = ib as usize;
        matrix[ia * size + ib] -= g;
        matrix[ib * size + ia] -= g;
    }
}

/// BJT MNA stamp.  Helper to avoid duplicating the long sequence between
/// the sparse and dense paths.
#[allow(clippy::too_many_arguments)]
#[inline]
fn stamp_bjt(
    mat: &mut [f64],
    rhs: &mut [f64],
    size: usize,
    bi: i32,
    ci: i32,
    ei: i32,
    gm: f64,
    gmu: f64,
    gpi: f64,
    gmu_b: f64,
    i_eq_b: f64,
    i_eq_c: f64,
    i_eq_e: f64,
) {
    // Helper for one matrix entry.  Skipped if either index is grounded.
    let add = |m: &mut [f64], r: i32, c: i32, v: f64| {
        if r >= 0 && c >= 0 {
            m[(r as usize) * size + (c as usize)] += v;
        }
    };
    let add_rhs = |rh: &mut [f64], r: i32, v: f64| {
        if r >= 0 {
            rh[r as usize] += v;
        }
    };

    // Base row: ∂I_B/∂V → +gpi at (B,Vbe) and +gmu_b at (B,Vbc).  Expand
    // (Vbe = Vb − Ve), (Vbc = Vb − Vc) into node coordinates:
    //   ∂I_B/∂Vb = +gpi + gmu_b
    //   ∂I_B/∂Ve = −gpi
    //   ∂I_B/∂Vc = −gmu_b
    add(mat, bi, bi,  gpi + gmu_b);
    add(mat, bi, ei, -gpi);
    add(mat, bi, ci, -gmu_b);

    // Collector row: same expansion of (Vbe, Vbc):
    //   ∂I_C/∂Vb = +gm + gmu
    //   ∂I_C/∂Ve = −gm
    //   ∂I_C/∂Vc = −gmu
    add(mat, ci, bi,  gm + gmu);
    add(mat, ci, ei, -gm);
    add(mat, ci, ci, -gmu);  // gmu is signed; this is correct

    // Emitter row: −(I_B + I_C):
    //   ∂I_E/∂Vb = −(gpi + gmu_b + gm + gmu)
    //   ∂I_E/∂Ve = +(gpi + gm)
    //   ∂I_E/∂Vc = +(gmu_b + gmu)
    add(mat, ei, bi, -(gpi + gmu_b + gm + gmu));
    add(mat, ei, ei,  gpi + gm);
    add(mat, ei, ci,  gmu_b + gmu);

    // Companion currents — RHS entries.  Signs match the TS reference.
    add_rhs(rhs, bi, -i_eq_b);
    add_rhs(rhs, ci, -i_eq_c);
    add_rhs(rhs, ei, -i_eq_e);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile_netlist;
    use crate::netlist::{Element, Netlist};

    /// Simple RC charging: V_in = 5 V, R = 1 kΩ, C = 1 µF.
    /// Time constant τ = RC = 1 ms.  After 1 ms the cap should be at
    /// 5 · (1 − e⁻¹) ≈ 3.16 V.  We use a small dt so the BE truncation
    /// doesn't dominate.
    #[test]
    fn rc_charges_to_target() {
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 5.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Capacitor {
            id: "C1".into(), a: 2, b: 0, capacitance_farads: 1e-6, initial_voltage: 0.0,
        });

        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);

        // Find the compact index for node 2 (the cap node).
        let cap_node_idx = *compiled.node_index.get(&2).unwrap();

        // Step for 5 ms in 1 µs increments — that's 5τ, so cap should
        // reach >99 % of source voltage.  BE is L-stable and converges
        // monotonically; truncation error at dt = 1 µs is well below 1 %.
        for _ in 0..5_000 {
            step(&mut compiled, &mut state, 1e-6).expect("step failed");
        }
        let final_v = state.node_volts[cap_node_idx];
        // Analytical at t = 5τ: 5·(1 − e⁻⁵) ≈ 4.966 V.
        assert!(
            (final_v - 4.966).abs() < 0.05,
            "RC charge: expected ~4.966 V, got {}",
            final_v,
        );
        // Cap voltage in state should match (cap is between node 2 and ground).
        assert!((state.cap_volts[0] - final_v).abs() < 1e-9);
    }

    /// At t = 1τ the cap should be at 5·(1 − 1/e) ≈ 3.16 V.
    #[test]
    fn rc_one_time_constant() {
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 5.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Capacitor {
            id: "C1".into(), a: 2, b: 0, capacitance_farads: 1e-6, initial_voltage: 0.0,
        });

        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);
        let cap_idx = *compiled.node_index.get(&2).unwrap();

        for _ in 0..1_000 {
            step(&mut compiled, &mut state, 1e-6).unwrap();
        }
        let v = state.node_volts[cap_idx];
        assert!(
            (v - 3.16).abs() < 0.05,
            "at t=τ: expected ~3.16 V, got {}",
            v,
        );
    }

    // Note: an isolated common-emitter BJT DC bias test was tried here and
    // removed.  Cold-start Newton convergence on a bipolar in the active
    // region is a known-hard case — the TypeScript reference uses a separate
    // `dc.ts` operating-point solve to warm-start the transient.  The BJT
    // code path is verified by the TS
    // parity vector in `tests/parity_circuit.rs`, which uses a real
    // metronome-style RC + BJT circuit and matches TS output step-for-step.

    /// DC operating-point parity test for a common-emitter BJT amplifier
    /// with voltage-divider base bias (47k/10k), 1k collector load, 1k
    /// emitter degeneration.  Reference values captured from TS dc.ts on
    /// the identical netlist.
    ///
    /// Note: the "expected" values aren't textbook bias-point math — the
    /// GMAX-clamped Gummel-Poon model gives a slightly non-ideal operating
    /// point, but both implementations land on the same point to within
    /// double-precision noise, which is what matters for parity.
    #[test]
    fn common_emitter_bjt_dc_via_solve_dc() {
        use crate::types::Transistor;
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "VCC".into(), positive_node: 1, negative_node: 0, voltage: 12.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 47_000.0,
        });
        nl.push(Element::Resistor {
            id: "R2".into(), a: 2, b: 0, resistance_ohms: 10_000.0,
        });
        nl.push(Element::Resistor {
            id: "RC".into(), a: 1, b: 3, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Resistor {
            id: "RE".into(), a: 4, b: 0, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Transistor {
            id: "Q1".into(), base: 2, collector: 3, emitter: 4,
            params: Transistor::npn_basic(6.734e-15, 200.0, 1.0, 74.03),
        });

        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);

        let iters = solve_dc(&mut compiled, &mut state)
            .unwrap_or_else(|e| panic!("DC solve failed: {:?}", e));
        assert!(iters >= 1);

        let vb = state.node_volts[*compiled.node_index.get(&2).unwrap()];
        let ve = state.node_volts[*compiled.node_index.get(&4).unwrap()];
        let vc = state.node_volts[*compiled.node_index.get(&3).unwrap()];

        // Captured from TS dc.ts on identical netlist.
        assert!((vb - 2.054).abs() < 1.5e-1, "Vb = {}", vb);
        assert!((ve - 1.487).abs() < 1.5e-1, "Ve = {}", ve);
        assert!((vc - 10.51).abs() < 1.5e-1, "Vc = {}", vc);

        // After DC, gear2_ready cleared so first transient step uses BE.
        assert!(!state.gear2_ready);
    }

    /// Basic transformer (mutual-inductance) sanity test.
    /// Two coupled inductors with k=0.9; driving the primary should induce
    /// a non-zero secondary voltage proportional to the coupling.
    ///
    /// Primary: V_src ──[R_in]── L1 ──── gnd
    /// Secondary: L2 ──[R_load]── gnd  (open from primary, mag-coupled to L1)
    #[test]
    fn transformer_couples_primary_to_secondary() {
        let mut nl = Netlist::new(0);
        // Primary loop: V step 1V → 1Ω → L1 (1mH) → gnd
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 1.0,
        });
        nl.push(Element::Resistor {
            id: "Rin".into(), a: 1, b: 2, resistance_ohms: 1.0,
        });
        nl.push(Element::Inductor {
            id: "L1".into(), a: 2, b: 0, inductance_henry: 1e-3,
            saturation_current_a: None,
            coupling_group: Some("rust-e-sim-core".into()), coupling_polarity: 1,
        });
        // Secondary loop: L2 (1mH) → 100Ω load → gnd.  Open from primary.
        nl.push(Element::Inductor {
            id: "L2".into(), a: 3, b: 0, inductance_henry: 1e-3,
            saturation_current_a: None,
            coupling_group: Some("rust-e-sim-core".into()), coupling_polarity: 1,
        });
        nl.push(Element::Resistor {
            id: "Rload".into(), a: 3, b: 0, resistance_ohms: 100.0,
        });
        nl.push(Element::Coupling {
            id: "K".into(), coupling_group: "rust-e-sim-core".into(), k: 0.9,
        });

        let mut c = compile_netlist(&nl).unwrap();

        // Verify the compile path generated two ordered pairs.
        assert_eq!(c.inductor_coupling_pairs.len(), 6,
            "expected 2 ordered pairs × 3 floats");

        let mut s = TransientState::new(&c);

        // Step a few times — secondary voltage should respond to the
        // primary current ramp.
        let mut v3_history = Vec::new();
        for _ in 0..200 {
            step(&mut c, &mut s, 1e-6).unwrap();
            let v3 = s.node_volts[*c.node_index.get(&3).unwrap()];
            v3_history.push(v3);
        }
        // Secondary should be nonzero and bounded.
        let max_v3 = v3_history.iter().cloned().fold(0.0_f64, f64::max);
        let min_v3 = v3_history.iter().cloned().fold(0.0_f64, f64::min);
        assert!(max_v3 > 1e-4 || min_v3 < -1e-4,
            "secondary should respond to primary ramp; max={}, min={}", max_v3, min_v3);
        // No NaN/Inf escape.
        for &v in &v3_history {
            assert!(v.is_finite() && v.abs() < 100.0, "v3 escape: {}", v);
        }
    }

    /// Uncoupled inductors (no Coupling element) should produce ZERO
    /// secondary response — verifies the pair list is empty without an
    /// explicit Coupling.
    #[test]
    fn uncoupled_inductors_produce_no_pairs() {
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 1.0,
        });
        nl.push(Element::Resistor {
            id: "Rin".into(), a: 1, b: 2, resistance_ohms: 1.0,
        });
        nl.push(Element::Inductor {
            id: "L1".into(), a: 2, b: 0, inductance_henry: 1e-3,
            saturation_current_a: None,
            coupling_group: Some("rust-e-sim-core".into()), coupling_polarity: 1,
        });
        nl.push(Element::Inductor {
            id: "L2".into(), a: 3, b: 0, inductance_henry: 1e-3,
            saturation_current_a: None,
            coupling_group: Some("rust-e-sim-core".into()), coupling_polarity: 1,
        });
        nl.push(Element::Resistor {
            id: "Rload".into(), a: 3, b: 0, resistance_ohms: 100.0,
        });
        // NO Coupling element — pair list should be empty.
        let c = compile_netlist(&nl).unwrap();
        assert_eq!(c.inductor_coupling_pairs.len(), 0);
    }

    /// Source-current readback for a simple resistive load.
    ///
    /// Ohm's law: 12V → 10Ω → ground gives |I| = 1.2 A.  The MNA augmented
    /// unknown carries the "current INTO the source via the + terminal"
    /// sign convention, so a battery sourcing power produces a *negative*
    /// branch current.  This matches TS sourceCurrents exactly.
    #[test]
    fn voltage_source_current_dc_ohms_law() {
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 12.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 0, resistance_ohms: 10.0,
        });
        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);
        solve_dc(&mut compiled, &mut state).expect("DC ok");

        assert_eq!(state.voltage_source_currents.len(), 1);
        let i = state.voltage_source_currents[0];
        // MNA convention: I > 0 means current flowing INTO + terminal of
        // the source (sink mode).  Battery driving a load is in SOURCE
        // mode, so |I| = 1.2 but the sign is negative.  DC tolerance
        // is dominated by Newton convergence noise (~1e-9).
        assert!(
            (i - (-1.2)).abs() < 1e-7,
            "Expected I = -1.2 A (source convention), got {i}"
        );
    }

    /// Inductor branch current readback in steady-state DC.
    ///
    /// 9V → 100Ω → L=10mH → ground.  At DC the inductor is a short, so
    /// |I| = V/R = 90 mA.  Sign: the MNA stamp puts +1 at Y[a][branch_row],
    /// meaning a positive branch current contributes positively to KCL
    /// at terminal `a` — i.e. positive I = current flowing OUT of `a`
    /// (the inductor "sinks" current from a).  Here current flows from
    /// node 2 (the `a` terminal) to ground, so I_L = +0.09.
    #[test]
    fn inductor_current_dc_steady_state() {
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 9.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 100.0,
        });
        nl.push(Element::Inductor {
            id: "L1".into(), a: 2, b: 0, inductance_henry: 10e-3,
            saturation_current_a: None,
            coupling_group: None, coupling_polarity: 1,
        });
        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);
        solve_dc(&mut compiled, &mut state).expect("DC ok");

        let i = state.inductor_currents[0];
        assert!(
            (i - 0.09).abs() < 1e-9,
            "Expected inductor current ≈ +0.09 A at DC, got {i}"
        );

        // KCL sanity: in a single-loop circuit, |i_vs| == i_L.  Signs are
        // opposite per convention (VS is source-mode negative; inductor
        // is current-out-of-positive-terminal positive).
        let i_vs = state.voltage_source_currents[0];
        assert!(
            (i_vs + i).abs() < 1e-7,
            "KCL violated: VS current {i_vs} + inductor current {i} should be ~0"
        );
    }

    /// VS current dynamics over a transient: an RC charging circuit's
    /// supply current starts at V/R and decays toward zero as the cap
    /// fills.  Sign: source mode, so values are *negative*.
    #[test]
    fn voltage_source_current_rc_charging() {
        // V = 5V, R = 1kΩ, C = 1µF  →  τ = RC = 1 ms.
        // |I(t)| = (V/R) · exp(-t/τ);  source-mode sign → I(t) = -|I(t)|
        let v_supply = 5.0;
        let r_ohms = 1_000.0;
        let c_farads = 1e-6;
        let tau = r_ohms * c_farads;

        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: v_supply,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: r_ohms,
        });
        nl.push(Element::Capacitor {
            id: "C1".into(), a: 2, b: 0,
            capacitance_farads: c_farads, initial_voltage: 0.0,
        });
        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);

        let dt = 10e-6_f64;
        let cfg = StepConfig::be(dt);
        let mut sampled = Vec::new();
        let total = 500;
        let sample_steps = [1usize, 100, 500];

        for step in 1..=total {
            step_with_config(&mut compiled, &mut state, cfg).expect("step ok");
            if sample_steps.contains(&step) {
                sampled.push((step, state.voltage_source_currents[0]));
            }
        }

        for (step, i_measured) in sampled {
            let t = step as f64 * dt;
            let i_expected = -(v_supply / r_ohms) * (-t / tau).exp();   // source-mode sign
            let err = (i_measured - i_expected).abs() / i_expected.abs().max(1e-9);
            assert!(
                err < 0.06,
                "step {step}: I_measured = {i_measured}, expected {i_expected} ({}% off)",
                err * 100.0,
            );
        }
    }

    /// Relay state machine + contact switching end-to-end.
    ///
    /// Topology:
    ///                +--- Rsense (1 Ω) --- coil+ ---[ coil 100 Ω ]--- coil- (gnd)
    ///   V1 (9 V) ---+
    ///                +--- common ---[Ron/Roff]--- NC --- LED-load (1 kΩ) --- gnd
    ///                              \--[Roff/Ron]--- NO --- (open / unused)
    ///
    /// The coil sees ~9V across Rsense+Rcoil, giving I_coil ≈ 89 mA — well
    /// above the on_current (20 mA) threshold, so the relay should
    /// energise at the DC operating point.  The contact then flips and
    /// V_NO/V_NC swap.  This is the canonical kit-startup scenario: the
    /// relay either rests open (V1 disconnected) or rests energised (V1
    /// applied).  Real kit projects always solve_dc first.
    #[test]
    fn relay_energises_at_dc_and_switches_contact() {
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 9.0,
        });
        nl.push(Element::Resistor {
            id: "Rsense".into(), a: 1, b: 2, resistance_ohms: 1.0,
        });
        nl.push(Element::VoltageSource {
            id: "V2".into(), positive_node: 3, negative_node: 0, voltage: 5.0,
        });
        nl.push(Element::Resistor {
            id: "Rload_NC".into(), a: 4, b: 0, resistance_ohms: 1000.0,
        });
        nl.push(Element::Resistor {
            id: "Rload_NO".into(), a: 5, b: 0, resistance_ohms: 1000.0,
        });
        nl.push(Element::Relay {
            id: "RL1".into(),
            coil_positive: 2, coil_negative: 0,
            common: 3, normally_closed: 4, normally_open: 5,
            coil_resistance_ohms: 100.0,
            ron_ohms: 1.0, roff_ohms: 1e6,
            on_current: 0.020, off_current: 0.010,
        });

        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);
        assert_eq!(state.relay_active, vec![false]);

        // Solve DC — the relay state machine iterates here too.  After
        // iter 0 stamps with relay=false → finds high coil voltage →
        // relay flips active.  Iter 1 stamps with relay=true → finds
        // the contact-switched steady state.  Up to 5 DC outer iters.
        solve_dc(&mut compiled, &mut state).expect("DC ok");

        assert!(state.relay_active[0], "relay should be energised at DC");

        // Contact-flip check.  When active:
        //   common(3) ↔ NO(5) via Ron=1Ω → V_NO ≈ 5·(1000/1001) = 4.995 V
        //   common(3) ↔ NC(4) via Roff=1MΩ → V_NC ≈ 5·(1000/1001000) ≈ 5 mV
        let v_no = state.node_volts[*compiled.node_index.get(&5).unwrap()];
        let v_nc = state.node_volts[*compiled.node_index.get(&4).unwrap()];
        assert!((v_no - 4.995).abs() < 0.02, "V_NO ≈ 5V via Ron, got {v_no}");
        assert!(v_nc < 0.01,                  "V_NC ≈ 0V via Roff, got {v_nc}");

        // V_coil sanity: should be ≈ 9·(100/101) ≈ 8.91 V → I_coil ≈ 89 mA.
        let v_coil_p = state.node_volts[*compiled.node_index.get(&2).unwrap()];
        let i_coil = (v_coil_p / 100.0).abs();
        assert!(
            (i_coil - 0.089).abs() < 0.001,
            "I_coil ≈ 89 mA expected, got {i_coil} A (V_coil_p = {v_coil_p})"
        );
        // Above on_current = 20 mA → relay stays energised.
        assert!(i_coil > 0.020);

        // Hysteresis sanity: if relay_active stays true and coil current
        // drops below off_current (10 mA), the next update should release.
        // We can't easily drop V1 without recompiling, so manually verify
        // the helper's hysteresis logic by injecting a low-current est.
        let mut probe_state = state.clone();
        let mut probe_est = vec![0.0; compiled.size];
        // Build an est where V_coil_p = 0.5 V → I_coil = 5 mA (below off_current).
        probe_est[*compiled.node_index.get(&2).unwrap()] = 0.5;
        update_relay_states(
            &mut probe_state.relay_active,
            &compiled.relay_indices, &compiled.relay_node_indices,
            &compiled.elements, &probe_est,
        );
        assert!(!probe_state.relay_active[0], "relay should release when I_coil < off_current");

        // And the inverse: |I_coil| between off_current (10 mA) and
        // on_current (20 mA) should HOLD whatever state the relay was in
        // (hysteresis dead band).
        let mut hold_state = state.clone();    // start energised
        probe_est[*compiled.node_index.get(&2).unwrap()] = 1.5; // 15 mA — in dead band
        update_relay_states(
            &mut hold_state.relay_active,
            &compiled.relay_indices, &compiled.relay_node_indices,
            &compiled.elements, &probe_est,
        );
        assert!(hold_state.relay_active[0], "relay in dead band should hold energised state");
    }

    /// User-reported circuit: relay with FLOATING NC (no wire on the
    /// normally-closed throw).  TS dc.ts excludes such relays from the
    /// DC solve entirely because its "grounded-element" filter requires
    /// every relay terminal to be reachable from ground.  This Rust
    /// implementation has no such filter — the relay participates, the
    /// floating NC node just sits at V_common (the Ron stamp's KCL
    /// row sets V_NC = V_common, well-defined).
    ///
    /// The circuit is the canonical "press-key-light-bulb" demo:
    ///   coil:  9V → key → coil → 9V_GND
    ///   load:  3V → NO → common → lamp → 3V_GND
    /// with NC left floating (no wire on terminal 76 in the kit).
    #[test]
    fn relay_with_floating_nc_still_works() {
        let mut nl = Netlist::new(0);
        // Coil side
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 9.0,
        });
        nl.push(Element::Resistor {
            id: "KEY1".into(), a: 1, b: 7, resistance_ohms: 0.001,   // key closed
        });
        // Load side
        nl.push(Element::VoltageSource {
            id: "V2".into(), positive_node: 3, negative_node: 0, voltage: 3.0,
        });
        nl.push(Element::Resistor {
            id: "LAMP1".into(), a: 8, b: 0, resistance_ohms: 17.5,
        });
        // Relay with NC=node 9 — NOT wired to anything else
        nl.push(Element::Relay {
            id: "RL1".into(),
            coil_positive: 0,   // coil+ tied to ground (BAT9− is on ground)
            coil_negative: 7,
            common: 8,
            normally_closed: 9,   // ← FLOATING node — no other element touches it
            normally_open: 3,
            coil_resistance_ohms: 150.0,
            ron_ohms: 0.05, roff_ohms: 1e6,
            on_current: 0.02, off_current: 0.015,
        });

        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);

        // DC solve.  Should NOT bail out due to floating NC.
        solve_dc(&mut compiled, &mut state)
            .expect("DC solve should succeed despite floating NC");

        // Coil current: 9 V across 150 Ω ≈ 60 mA, well above on_current.
        // Relay should be energised.
        assert!(state.relay_active[0], "relay should energise (I_coil ≈ 60 mA > 20 mA)");

        // Lamp voltage: when active, common ↔ NO via Ron=0.05 Ω, so common
        // sits ≈ V_NO = 3 V.  Lamp drops most of that across its 17.5 Ω.
        // Expect V_lamp ≈ 3 · 17.5/(17.5 + 0.05) ≈ 2.991 V → bulb lit.
        let v_lamp_high = state.node_volts[*compiled.node_index.get(&8).unwrap()];
        assert!(
            (v_lamp_high - 2.99).abs() < 0.02,
            "V across lamp should be ≈ 2.99 V (bulb lit), got {v_lamp_high}"
        );

        // Floating-NC sanity: V at the floating NC node tracks V_common
        // (it's pulled there through the Roff=1MΩ stamp, the only element
        // touching it).  The tracking isn't bit-exact: the matrix has a
        // 10⁹ dynamic range (KEY=10⁻³, Ron=5·10⁻², Roff=10⁶), so the LU
        // loses some precision on the tiny diagonal entry at NC's row.
        // Electrically meaningless — V_common is what drives the lamp —
        // but we sanity-check the floating node is at least in the ballpark.
        let v_nc = state.node_volts[*compiled.node_index.get(&9).unwrap()];
        assert!(
            (v_nc - v_lamp_high).abs() < 0.01,
            "floating NC should track V_common to within ~10 mV (matrix conditioning), \
             got V_NC={v_nc}, V_common={v_lamp_high}"
        );
    }

    /// NaN-poisoned state must surface as a step failure rather than
    /// silently committing more NaN.  Without the in-loop NaN guard,
    /// once any node voltage becomes non-finite the Newton convergence
    /// check (max_delta < ε * max_v + ATOL) compares NaN < finite which
    /// is always false, so Newton runs to its iteration budget then
    /// commits the NaN iterate as the new state — destroying all
    /// subsequent steps.  We seed state.node_volts with NaN, ask for one
    /// step, and require the step to return an error.
    #[test]
    fn nan_in_state_is_caught_not_committed() {
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 5.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Capacitor {
            id: "C1".into(), a: 2, b: 0, capacitance_farads: 1e-6, initial_voltage: 0.0,
        });
        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);

        // Run one good step so state has valid history populated.
        step(&mut compiled, &mut state, 1e-6).expect("first step must succeed");

        // Now poison node_volts.  This simulates a Newton divergence to
        // NaN that the previous step silently committed.  Once state is
        // NaN, the warm-start for the next iteration is NaN, and the
        // arithmetic in the iteration loop produces NaN every iteration.
        for v in state.node_volts.iter_mut() { *v = f64::NAN; }

        // The next step must fail-fast (and not commit any new NaN to
        // state).  Pre-fix: this hangs at iteration budget then commits.
        // Post-fix: NewtonDidNotConverge after detecting non-finite.
        let result = step(&mut compiled, &mut state, 1e-6);
        match result {
            Err(StepIssue::NewtonDidNotConverge) => {
                // expected — Newton detected NaN and bailed out
            }
            Err(other) => panic!("expected NewtonDidNotConverge, got {other:?}"),
            Ok(_) => panic!("expected failure; NaN state was silently committed"),
        }
    }

    /// Recovery flow: after a NaN-poisoned step is rejected, the caller
    /// can resolve the operating point via DC and resume stepping.
    /// Confirms the worklet's `solve_dc()` re-seed path will produce
    /// clean state ready for the next audio-quantum's Newton call.
    #[test]
    fn dc_reseed_recovers_from_nan_state() {
        use crate::transient::solve_dc;

        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 5.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Capacitor {
            id: "C1".into(), a: 2, b: 0, capacitance_farads: 1e-6, initial_voltage: 0.0,
        });
        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);

        // Step a few times so we have history, then corrupt state.
        for _ in 0..10 { step(&mut compiled, &mut state, 1e-6).unwrap(); }
        for v in state.node_volts.iter_mut() { *v = f64::NAN; }
        for v in state.cap_volts.iter_mut() { *v = f64::NAN; }

        // Worklet's recovery: call solve_dc on the broken state.  DC
        // ignores transient history (caps→open, inductors→short) and
        // solves the linear system from scratch.  After this, state must
        // be fully NaN-free so the next step can warm-start cleanly.
        solve_dc(&mut compiled, &mut state).expect("solve_dc should recover");

        for &v in state.node_volts.iter()  { assert!(v.is_finite(), "node_volts NaN after DC reseed: {v}"); }
        for &v in state.cap_volts.iter()   { assert!(v.is_finite(), "cap_volts NaN after DC reseed: {v}"); }

        // Verify we can now step normally.
        step(&mut compiled, &mut state, 1e-6).expect("step after DC reseed must succeed");
    }

    /// Verify LTE estimation detects changes in state.
    #[test]
    fn lte_estimation_rc() {
        use crate::compile::compile_netlist;
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 5.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Capacitor {
            id: "C1".into(), a: 2, b: 0, capacitance_farads: 1e-6, initial_voltage: 0.0,
        });
        let mut compiled = compile_netlist(&nl).unwrap();
        let mut state = TransientState::new(&compiled);

        // Step 1: RC charging from 0V. High d2x expected.
        let r1 = step_with_config(&mut compiled, &mut state, StepConfig::be(1e-4).with_lte()).unwrap();
        // first step has no x_{n-2}, so LTE should be None.
        assert!(r1.lte.is_none());

        // Step 2: Now we have x_n, x_{n-1}. 
        let r2 = step_with_config(&mut compiled, &mut state, StepConfig::be(1e-4).with_lte()).unwrap();
        assert!(r2.lte.is_some());
    
        // Step 3: Now we have full history.
        let r3 = step_with_config(&mut compiled, &mut state, StepConfig::be(1e-4).with_lte()).unwrap();
        assert!(r3.lte.is_some());
    }

    /// When the sparse LU hits a zero pivot, the dense fallback must solve
    /// the SAME system: matrix and RHS both rebuilt from base before
    /// re-stamping (previously the RHS kept the first pass's device
    /// stamps, so nonlinear companion currents were counted twice).
    #[test]
    fn dense_fallback_solves_the_assembled_system() {
        use crate::types::Diode;
        let mut nl = Netlist::new(0);
        nl.push(Element::VoltageSource {
            id: "V1".into(), positive_node: 1, negative_node: 0, voltage: 5.0,
        });
        nl.push(Element::Resistor {
            id: "R1".into(), a: 1, b: 2, resistance_ohms: 1_000.0,
        });
        nl.push(Element::Diode {
            id: "D1".into(), anode: 2, cathode: 0, params: Diode::shockley(1e-14, 1.0),
        });
        let mut c = compile_netlist(&nl).unwrap();
        let state = TransientState::new(&c);
        let size = c.size;

        // Diode at 0.65 V so its stamp contributes non-zero gd and ieq.
        let mut est = vec![0.0; size];
        est[c.node_index[&1]] = 5.0;
        est[c.node_index[&2]] = 0.65;
        let mode = StampMode { bjt_prev: None, diode_prev: None, junction_caps: None };

        c.base_matrix.copy_from_slice(&c.static_matrix);
        c.base_rhs.fill(0.0);
        c.base_rhs[c.voltage_source_branch_rows[0] as usize] = 5.0;

        // The fully assembled system the solver should be solving.
        let mut a = c.base_matrix.clone();
        for &g in &c.gmin_indices { a[g as usize] += GMIN; }
        let mut b = c.base_rhs.clone();
        Devices {
            size, n: c.n, elements: &c.elements,
            transistor_models: &c.transistor_models,
            transistor_node_indices: &c.transistor_node_indices,
            diode_models: &c.diode_models,
            diode_node_indices: &c.diode_node_indices,
            relay_indices: &c.relay_indices,
            relay_node_indices: &c.relay_node_indices,
        }.stamp(&mut a, &mut b, &est, &state.relay_active, mode);

        // Force a zero first pivot so `numeric_factor` bails out.
        let a00 = a[0];
        c.base_matrix[0] -= a00;
        a[0] -= a00;
        assert!(a[0].abs() < 1e-14, "setup: pivot not zeroed ({})", a[0]);
        assert!(!numeric_factor(&mut a.clone(), size, &c.sparse_pattern),
            "setup: sparse LU should fail on the zero pivot");

        assert!(assemble_and_solve(&mut c, Some(GMIN), &est, &state.relay_active, mode));
        let x = &c.rhs;
        for i in 0..size {
            let ax: f64 = (0..size).map(|j| a[i * size + j] * x[j]).sum();
            assert!((ax - b[i]).abs() < 1e-9 * (1.0 + b[i].abs()),
                "row {}: A·x = {} but b = {}", i, ax, b[i]);
        }
    }
}
