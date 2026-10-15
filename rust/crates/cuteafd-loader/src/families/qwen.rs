//! Geometry shared by dense Qwen and Flash Next, independent of mHC/MoE/QSA.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridAttention {
    Gdn,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnOutputGate {
    Sigmoid,
    Silu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnGeometry {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub output_gate: GdnOutputGate,
}

impl GdnGeometry {
    pub fn conv_width(self) -> usize {
        2 * self.key_heads * self.key_dim + self.value_width()
    }

    pub fn value_width(self) -> usize {
        self.value_heads * self.value_dim
    }

    /// Exact FP32 recurrent matrix plus the BF16 causal-convolution tail.
    pub fn state_bytes(self) -> usize {
        self.value_heads * self.key_dim * self.value_dim * 4
            + (self.conv_kernel - 1) * self.conv_width() * 2
    }
}
