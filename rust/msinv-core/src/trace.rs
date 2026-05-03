//! Item-2 perf investigation: env-gated trace of |active| over time.
//!
//! Activated by setting `MSINV_TRACE_ACTIVE=path/to/trace.tsv` at process
//! start. Output is appended to that file. Zero overhead when unset
//! (single atomic load + branch via `OnceLock`).
//!
//! Schema (tab-separated, no header):
//!   ITER\t<t>\t<n_active>\t<n_a_tag>\t<phase>\t<event>
//!   MARK\t<t>\t<kind>\t<k>=<v>...
//!
//! `phase` is one of:
//!   N — neutral (no active sweep)
//!   S — selection phase (tau <= t < t_origin, going backward)
//!   V — standing-variation phase (t_origin <= t < t_de_novo)
//!
//! Used by `.tmp/analyze_active_trace.py` to characterise growth pattern
//! during sweep windows.
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

static WRITER: OnceLock<Option<Mutex<BufWriter<std::fs::File>>>> = OnceLock::new();

fn writer() -> Option<&'static Mutex<BufWriter<std::fs::File>>> {
    let slot = WRITER.get_or_init(|| match std::env::var("MSINV_TRACE_ACTIVE") {
        Ok(p) if !p.is_empty() => match OpenOptions::new().append(true).create(true).open(&p) {
            Ok(f) => {
                let mut bw = BufWriter::new(f);
                let _ = writeln!(bw, "# msinv trace pid={}", std::process::id());
                Some(Mutex::new(bw))
            }
            Err(_) => None,
        },
        _ => None,
    });
    slot.as_ref()
}

#[inline(always)]
pub fn enabled() -> bool {
    writer().is_some()
}

#[inline]
pub fn iter(t: f64, n_active: usize, n_a_tag: usize, phase: char, event: &str) {
    let Some(m) = writer() else { return };
    if let Ok(mut bw) = m.lock() {
        let _ = writeln!(
            bw,
            "ITER\t{:.6e}\t{}\t{}\t{}\t{}",
            t, n_active, n_a_tag, phase, event
        );
    }
}

#[inline]
pub fn mark(t: f64, kind: &str, kvs: &[(&str, &str)]) {
    let Some(m) = writer() else { return };
    if let Ok(mut bw) = m.lock() {
        let _ = write!(bw, "MARK\t{:.6e}\t{}", t, kind);
        for (k, v) in kvs {
            let _ = write!(bw, "\t{}={}", k, v);
        }
        let _ = writeln!(bw);
    }
}

pub fn flush() {
    let Some(m) = writer() else { return };
    if let Ok(mut bw) = m.lock() {
        let _ = bw.flush();
    }
}
