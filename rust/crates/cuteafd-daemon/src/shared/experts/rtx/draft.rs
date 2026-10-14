//! Capture-compatible top-3 native draft experts and ordered rank-pair reduction.
//! Unlike backbone owner reduction, each route sums ranks in FP32 and rounds to
//! BF16 before the top-3 sum and optional shared contribution.
pub(crate) mod pair;
pub(crate) mod rank;

/// Checkpoint draft shape and caller-selected owner rank (not a physical GPU id).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub geometry: cuteafd_core::ExpertGeometry,
    pub stages: usize,
    pub owner: usize,
}
impl Layout {
    pub fn validate(self) -> anyhow::Result<()> {
        let g = self.geometry;
        anyhow::ensure!(
            g.hidden == 5120
                && g.experts == 128
                && g.topk == 3
                && g.intermediate == 2304
                && self.stages == 3
                && g.layers as usize == self.stages,
            "native draft ABI does not support geometry {} / {} stages",
            g.key(),
            self.stages
        );
        anyhow::ensure!(self.owner < 2, "draft owner rank is out of range");
        Ok(())
    }
    pub fn input_bytes(self) -> [usize; 3] {
        let hidden = self.geometry.hidden as usize;
        [
            hidden + hidden / 32,
            self.geometry.topk as usize * 4,
            self.geometry.topk as usize * 4,
        ]
    }
    pub fn route_bytes(self) -> usize {
        self.geometry.topk as usize * self.geometry.hidden as usize * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn layout() -> Layout {
        Layout {
            geometry: cuteafd_core::ExpertGeometry {
                hidden: 5120,
                experts: 128,
                topk: 3,
                intermediate: 2304,
                layers: 3,
            },
            stages: 3,
            owner: 1,
        }
    }
    #[test]
    fn draft_layout_derives_extents_and_rejects_unsupported_native_abi() {
        let mut l = layout();
        assert!(l.validate().is_ok());
        assert_eq!(l.input_bytes(), [5280, 12, 12]);
        assert_eq!(l.route_bytes(), 3 * 5120 * 4);
        l.owner = 0;
        assert!(l.validate().is_ok());
        l.owner = 2;
        assert!(l.validate().is_err());
        for geometry in [
            cuteafd_core::ExpertGeometry {
                hidden: 4096,
                ..layout().geometry
            },
            cuteafd_core::ExpertGeometry {
                topk: 6,
                ..layout().geometry
            },
            cuteafd_core::ExpertGeometry {
                experts: 384,
                ..layout().geometry
            },
            cuteafd_core::ExpertGeometry {
                intermediate: 1152,
                ..layout().geometry
            },
            cuteafd_core::ExpertGeometry {
                layers: 4,
                ..layout().geometry
            },
        ] {
            assert!(Layout {
                geometry,
                ..layout()
            }
            .validate()
            .is_err());
        }
        assert!(Layout {
            stages: 4,
            ..layout()
        }
        .validate()
        .is_err());
    }
}
