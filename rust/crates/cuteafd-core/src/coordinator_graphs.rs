use serde::{Deserialize, Serialize};

use crate::{
    DType, CuteafdError, GraphBucket, LayerWaveMode,
};

pub const COORDINATOR_GRAPH_DECODE_BUCKET_ROWS: usize = 1;
pub const COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS: [usize; 13] = [
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
pub const COORDINATOR_GRAPH_SHAPES: [CoordinatorGraphShape; 5] = [
    CoordinatorGraphShape::CoordAttention,
    CoordinatorGraphShape::CoordCompressedAttention,
    CoordinatorGraphShape::CoordDense,
    CoordinatorGraphShape::CoordSparseA,
    CoordinatorGraphShape::CoordSparseB,
];
pub const COORDINATOR_GRAPH_INSTANCE_COUNT: usize =
    COORDINATOR_GRAPH_SHAPES.len() * (1 + COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS.len());

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoordinatorGraphShape {
    CoordAttention,
    CoordCompressedAttention,
    CoordDense,
    CoordSparseA,
    CoordSparseB,
}

impl CoordinatorGraphShape {
    pub fn label(self) -> &'static str {
        match self {
            Self::CoordAttention => "Coord-Attention",
            Self::CoordCompressedAttention => "Coord-Compressed-Attention",
            Self::CoordDense => "Coord-Dense",
            Self::CoordSparseA => "Coord-Sparse-A",
            Self::CoordSparseB => "Coord-Sparse-B",
        }
    }

    pub fn op_count(self) -> usize {
        match self {
            Self::CoordAttention | Self::CoordCompressedAttention => 1,
            Self::CoordDense => 14,
            Self::CoordSparseA => 12,
            Self::CoordSparseB => 2,
        }
    }

    pub fn network_boundary(self) -> CoordinatorGraphNetworkBoundary {
        match self {
            Self::CoordAttention | Self::CoordCompressedAttention => {
                CoordinatorGraphNetworkBoundary::None
            }
            Self::CoordDense => CoordinatorGraphNetworkBoundary::None,
            Self::CoordSparseA => CoordinatorGraphNetworkBoundary::BeforeExpertSend,
            Self::CoordSparseB => CoordinatorGraphNetworkBoundary::AfterExpertRecv,
        }
    }


}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoordinatorGraphNetworkBoundary {
    None,
    BeforeExpertSend,
    AfterExpertRecv,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinatorGraphKey {
    pub shape: CoordinatorGraphShape,
    pub row_bucket: GraphBucket,
    pub dtype: DType,
}

impl CoordinatorGraphKey {

    pub fn new(
        shape: CoordinatorGraphShape,
        _mode: LayerWaveMode,
        active_rows: usize,
        dtype: DType,
    ) -> Result<Self, CuteafdError> {
        Ok(Self {
            shape,
            row_bucket: coordinator_graph_bucket_for_active_rows(active_rows)?,
            dtype,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinatorGraphInstancePlan {
    pub key: CoordinatorGraphKey,
    pub op_count: usize,
    pub network_boundary: CoordinatorGraphNetworkBoundary,
}

impl CoordinatorGraphInstancePlan {

    fn new(key: CoordinatorGraphKey, shape: CoordinatorGraphShape) -> Self {
        Self {
            key,
            op_count: shape.op_count(),
            network_boundary: shape.network_boundary(),
        }
    }
}

pub fn coordinator_graph_bucket_for_active_rows(
    active_rows: usize,
) -> Result<GraphBucket, CuteafdError> {
    if active_rows == 0 {
        return Err(CuteafdError::GraphBufferContractInvalid {
            reason: "coordinator graph active rows must be nonzero".to_owned(),
        });
    }
    if active_rows == COORDINATOR_GRAPH_DECODE_BUCKET_ROWS {
        return Ok(GraphBucket::decode());
    }
    for row_capacity in COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS {
        if active_rows <= row_capacity {
            return Ok(GraphBucket::new(row_capacity));
        }
    }
    Err(CuteafdError::GraphBufferContractInvalid {
        reason: format!(
            "coordinator graph active rows {} exceed max prefill bucket {}",
            active_rows,
            COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS[COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS.len() - 1]
        ),
    })
}
