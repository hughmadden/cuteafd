//! Routed draft compute with fixed inputs/output and lane-owned graph storage.
use super::{DsparkRouter, DsparkSharedFfn};
use crate::families::deepseek_v41::v41_experts::ExpertExecution;
use anyhow::Result;
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::ffi::c_void;

pub(super) enum DraftExperts<'w, 'a> {
    Tp2(super::tp2::RoutedWave<'w, 'a>),
    Full(ExpertExecution<'w, 'a>),
}
impl DraftExperts<'_, '_> {
    pub fn stream(&self) -> *mut c_void {
        match self {
            Self::Full(v) => v.stream(),
            Self::Tp2(v) => v.stream(),
        }
    }
    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Full(v) => v.synchronize(),
            Self::Tp2(v) => v.synchronize(),
        }
    }
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 3] {
        match self {
            Self::Full(v) => v.inputs(),
            Self::Tp2(v) => v.inputs(),
        }
    }
    pub fn output(&self) -> Option<CuteafdDeviceBuffer> {
        match self {
            Self::Full(v) => v.output(),
            Self::Tp2(v) => Some(v.output()),
        }
    }
    pub unsafe fn enqueue_draft_ffn_on(
        &mut self,
        router: &mut DsparkRouter<'_, '_>,
        shared: &mut DsparkSharedFfn<'_, '_>,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        match self {
            Self::Full(v) => unsafe { v.enqueue_draft_ffn_on(router, shared, rows, stream) },
            Self::Tp2(v) => unsafe { v.enqueue(router, shared, rows, stream) },
        }
    }
}
