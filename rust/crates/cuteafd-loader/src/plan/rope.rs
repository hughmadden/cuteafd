//! Load-time rotary-table recipes. Families select policy; storage is shared.
use anyhow::{ensure, Context, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3 {
    pub factor: f32,
    pub original_max: f32,
    pub low_factor: f32,
    pub high_factor: f32,
}

impl Llama3 {
    pub fn parse(value: &serde_json::Value) -> Result<Self> {
        ensure!(
            value["rope_type"] == "llama3" || value["type"] == "llama3",
            "native rotary tables require llama3 rope_scaling"
        );
        let number = |name: &str| -> Result<f32> {
            let n = value[name]
                .as_f64()
                .context(format!("rope_scaling requires {name}"))?;
            let f = n as f32;
            ensure!(
                n.is_finite() && f.is_finite() && f > 0.0,
                "rope_scaling.{name} must be positive finite FP32"
            );
            Ok(f)
        };
        let result = Self {
            factor: number("factor")?,
            original_max: number("original_max_position_embeddings")?,
            low_factor: number("low_freq_factor")?,
            high_factor: number("high_freq_factor")?,
        };
        ensure!(
            result.high_factor > result.low_factor,
            "llama3 requires high_freq_factor > low_freq_factor"
        );
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recipe {
    pub span: usize,
    pub theta: f32,
    pub scaling: Option<Llama3>,
}

impl Recipe {
    pub fn new(span: usize, theta: f64, scaling: Option<Llama3>) -> Result<Self> {
        ensure!(
            span > 0 && span % 2 == 0,
            "rotary span must be positive and even"
        );
        let theta = theta as f32;
        ensure!(
            theta.is_finite() && theta > 0.0,
            "rotary theta must be positive finite FP32"
        );
        Ok(Self {
            span,
            theta,
            scaling,
        })
    }

    pub fn program(&self) -> String {
        format!(
            "rope_s{}_{}",
            self.span,
            if self.scaling.is_some() {
                "llama3"
            } else {
                "plain"
            }
        )
    }

    /// Two BF16 planes, each charged at the allocator's 256-byte floor.
    pub fn allocation_bytes(&self, extent: usize) -> Result<u64> {
        ensure!(
            extent > 0 && extent <= i32::MAX as usize,
            "invalid rotary extent"
        );
        let plane = (extent as u64)
            .checked_mul(self.span as u64)
            .context("rotary table size overflow")?
            .max(256);
        plane.checked_mul(2).context("rotary table size overflow")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tables {
    pub extent: usize,
    pub recipes: Vec<Recipe>,
    pub allocation_bytes: u64,
}

impl Tables {
    pub fn new(extent: usize, recipes: impl IntoIterator<Item = Recipe>) -> Result<Self> {
        let mut unique = Vec::new();
        let mut allocation_bytes = 0u64;
        for recipe in recipes {
            if !unique.contains(&recipe) {
                allocation_bytes = allocation_bytes
                    .checked_add(recipe.allocation_bytes(extent)?)
                    .context("rotary tables size overflow")?;
                unique.push(recipe);
            }
        }
        ensure!(!unique.is_empty(), "rotary table plan has no recipes");
        Ok(Self {
            extent,
            recipes: unique,
            allocation_bytes,
        })
    }

    pub fn check_budget(&self, available: u64, reservations: &[u64]) -> Result<()> {
        let required = reservations
            .iter()
            .try_fold(self.allocation_bytes, |n, bytes| n.checked_add(*bytes))
            .context("rotary reservation overflow")?;
        ensure!(
            required <= available,
            "rotary tables require {required} bytes, available {available}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_deduplicate_full_recipes_and_charge_allocator_floors() -> Result<()> {
        let recipe = Recipe::new(32, 10000.0, None)?;
        let other = Recipe::new(32, 10000000.0, None)?;
        let tables = Tables::new(1, [recipe, recipe, other])?;
        assert_eq!(tables.recipes, vec![recipe, other]);
        assert_eq!(tables.allocation_bytes, 1024);
        assert!(tables.check_budget(1023, &[]).is_err());
        assert!(tables.check_budget(1024, &[1]).is_err());
        assert!(tables.check_budget(u64::MAX, &[u64::MAX]).is_err());
        tables.check_budget(1024, &[])?;
        assert!(Tables::new(0, [recipe]).is_err());
        assert!(Recipe::new(31, 10000.0, None).is_err());
        assert!(Recipe::new(32, f64::INFINITY, None).is_err());
        Ok(())
    }
}
