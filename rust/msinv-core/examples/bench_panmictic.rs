use std::env;
use std::time::Instant;

use msinv_core::simulator::HullSimulator;

fn main() {
    let args: Vec<String> = env::args().collect();
    let rho: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000.0);
    let reps: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let ne: f64 = 1000.0;
    let l: f64 = 100_000.0;
    let r = rho / (4.0 * ne * l);

    println!("panmictic rho={rho} reps={reps} n=20 Ne={ne} L={l}");
    let t0 = Instant::now();
    for rep in 0..reps {
        let sim = HullSimulator::simple(
            20, 0, ne, l, r, vec![], 3000 + rep);
        let _result = sim.simulate();
    }
    let total = t0.elapsed().as_secs_f64();
    println!("total {:.3} s  mean {:.3} s/rep  {:.1} reps/s",
             total, total / reps as f64, reps as f64 / total);
}
