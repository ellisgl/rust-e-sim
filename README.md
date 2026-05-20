# rust-e-sim

High-performance, purely functional circuit simulation kernels in Rust.

`rust-e-sim` is a professional-grade circuit simulation engine designed for low-latency transient analysis. It provides the core numerical routines required for SPICE-like simulation, specifically optimized for WebAssembly integration with modern reactive frameworks like Svelte 5.

## Features

- **Modified Nodal Analysis (MNA)**: Robust matrix formulation for circuit topology.
- **Sparse LU Solver**: Optimized linear algebra for small-to-medium circuits (N ≤ 100).
- **Newton-Raphson Iteration**: Convergent solver for nonlinear devices (diodes, transistors).
- **Adaptive Integration**:
    - **Backward Euler**: Robust first-order integration.
    - **BDF-2 (Gear-2)**: High-accuracy second-order integration.
- **Advanced Semiconductor Models**:
    - **Gummel-Poon BJT**: Standard transistor model with high-injection and Early effect.
    - **Shockley Diode**: Standard diode model with Zener breakdown.
    - **SPICE pnjlim**: Voltage limiting for numerical stability in nonlinear regions.
- **State Persistence**: Full serialization/deserialization of simulation state via `serde`.
- **High-Performance WASM Bridge**: Minimal-copy interface for browser-based simulation.

## Architecture

The project is split into two main components:

### `rust-e-sim-core`
The pure-Rust simulation engine. It is `no_std`-compatible (with `alloc`) and handles:
- Netlist compilation and node reordering (Minimum Degree).
- Sparse and dense linear solvers.
- Nonlinear element stamping.
- Transient time-stepping logic.

### `rust-e-sim-wasm`
A `wasm-bindgen` wrapper that exposes the core engine to JavaScript/TypeScript.
- Optimized for the browser's `AudioWorklet` or main thread.
- Provides a `Simulator` class with a simple API for building and stepping circuits.
- Supports seamless "pot-turn" scenarios where component values are updated without resetting the simulation progress.

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (for building the WASM bridge)

### Building

To build the core library:
```bash
cargo build -p rust-e-sim-core
```

To build the WASM bridge:
```bash
cd rust-e-sim-wasm
wasm-pack build --target web
```

### Testing

The project includes a comprehensive test suite covering numerical parity and physical correctness:
```bash
cargo test --all-targets
```

## Integration with Svelte 5

`rust-e-sim` is designed to work seamlessly with Svelte 5 Runes. You can wrap the `Simulator` state in a `$state` rune and use `get_full_state()` / `set_full_state()` for persistent session management.

```typescript
import init, { Simulator } from './pkg/rust_e_sim_wasm.js';

await init();
const sim = new Simulator(0); // Ground node ID = 0
sim.add_voltage_source("V1", 1, 0, 5.0);
sim.add_resistor("R1", 1, 2, 1000.0);
sim.add_capacitor("C1", 2, 0, 1e-6, 0.0);

if (sim.compile()) {
  const result = sim.step(1e-6); // Step 1 microsecond
  console.log("Node 2 voltage:", sim.node_voltage(2));
}
```

## License

MIT OR Apache-2.0
