//! Admitted TP2 expert halves whose weights land before the engine exists.
//! Weight owners carry no execution state (modules, workspaces, routes), so
//! they can load inside the Spark admission window; execution binds after the
//! engine is created and before any step workspace is allocated.
use super::{exl3::{Exl3Tp2, Exl3Tp2Weights}, fp8moe::Fp8MoeTp2, RtxExpertLayer};
use anyhow::Result;

pub(crate) enum Tp2Weights<'a> {
    Exl3([Exl3Tp2Weights<'a>; 2]),
    /// FP8/NVFP4 halves keep their existing owner: its executor holds no
    /// engine, pool or transport state.
    Fp8([Fp8MoeTp2<'a>; 2]),
}

impl<'a> Tp2Weights<'a> {
    pub(crate) fn execution(self) -> Result<[Box<dyn RtxExpertLayer + 'a>; 2]> {
        Ok(match self {
            Self::Exl3([left, right]) => {
                let pair: [Exl3Tp2<'a>; 2] = [left.execution()?, right.execution()?];
                pair.map(|rank| Box::new(rank) as Box<dyn RtxExpertLayer>)
            }
            Self::Fp8(pair) => pair.map(|rank| Box::new(rank) as Box<dyn RtxExpertLayer>),
        })
    }
}
