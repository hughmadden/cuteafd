//! One serialized TP2 expert rank retains package scratch and a BF16 output.
use anyhow::{ensure, Context, Result};

pub fn fp8moe_tp2_workspace(scratch: usize, hidden: usize, rows: usize) -> Result<(usize, usize, usize)> {
    ensure!(rows > 0 && hidden > 0, "invalid FP8 MoE TP2 output geometry");
    let output = rows.checked_mul(hidden).and_then(|n| n.checked_mul(2))
        .context("FP8 MoE TP2 output size overflow")?;
    let scratch = scratch.max(256);
    let total = scratch.checked_add(output).context("FP8 MoE TP2 workspace size overflow")?;
    Ok((scratch, output, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_floor_and_overflow() {
        assert_eq!(fp8moe_tp2_workspace(0, 6144, 64).unwrap(), (256, 786432, 786688));
        assert!(fp8moe_tp2_workspace(1, 0, 64).is_err());
        assert!(fp8moe_tp2_workspace(usize::MAX, 6144, 64).is_err());
        assert!(fp8moe_tp2_workspace(1, usize::MAX, 64).is_err());
    }
}
