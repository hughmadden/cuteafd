//! CPU-only EXL3 semantic image, addressed after the device prefix in a push
//! rank's logical arena. It is never exposed to kernels or copied to the GPU.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{
    push_load::builder::{Builder, Destination},
    OfficialV41Catalog, V41Exl3Residency,
};

pub(crate) struct ValidationLayout {
    names: Vec<String>,
    base: u64,
}
impl ValidationLayout {
    pub fn exl3(
        residencies: &[V41Exl3Residency],
        base: u64,
        admitted_host_bytes: usize,
    ) -> Result<Self> {
        let names = residencies
            .iter()
            .flat_map(|r| &r.loads)
            .filter_map(|load| {
                load.tensor
                    .strip_suffix(".trellis")
                    .map(|prefix| format!("{prefix}.mcg"))
            })
            .collect::<Vec<_>>();
        let bytes = names
            .len()
            .checked_mul(4)
            .context("EXL3 validation size overflow")?;
        ensure!(
            bytes <= admitted_host_bytes && base.checked_add(bytes as u64).is_some(),
            "EXL3 validation image exceeds admission"
        );
        Ok(Self { names, base })
    }
    pub fn bytes(&self) -> usize {
        self.names.len() * 4
    }
    pub fn base(&self) -> u64 {
        self.base
    }
    pub fn append(
        &self,
        builder: &mut Builder,
        catalog: &OfficialV41Catalog,
        rank: u32,
    ) -> Result<()> {
        for (index, name) in self.names.iter().enumerate() {
            let source = Builder::v41_source(catalog, name)?;
            ensure!(source.bytes == 4, "EXL3 multiplier is not a scalar");
            builder.contiguous(
                &source,
                0,
                4,
                Destination {
                    rank,
                    offset: self.base + index as u64 * 4,
                    bytes: 4,
                },
            )?;
        }
        Ok(())
    }
    pub fn validate(&self, image: &[u8]) -> Result<()> {
        ensure!(
            image.len() == self.bytes(),
            "EXL3 validation image length mismatch"
        );
        for (name, bytes) in self.names.iter().zip(image.chunks_exact(4)) {
            ensure!(
                u32::from_le_bytes(bytes.try_into().unwrap()) == 0xcbac1fed,
                "unsupported EXL3 MCG multiplier for {name}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validation_image_is_exact_and_semantic_not_a_checksum() -> Result<()> {
        let layout = ValidationLayout {
            names: vec!["projection.mcg".into()],
            base: 4096,
        };
        assert!(layout.validate(&[0; 4]).is_err());
        assert!(layout.validate(&[]).is_err());
        layout.validate(&0xcbac1fedu32.to_le_bytes())?;
        assert_eq!(layout.base(), 4096);
        Ok(())
    }
}
