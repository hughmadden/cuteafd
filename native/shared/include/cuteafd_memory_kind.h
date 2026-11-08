#pragma once
// What one allocation holds (Rust cuteafd_core::memory_layout::MemoryKind).
// Every allocation entry point takes a kind and a nonempty owner name; 0 is
// never a kind. CUDA context, modules and library workspaces are a measured
// residual, not a kind.
typedef enum cuteafd_memory_kind_t {
  CUTEAFD_MEMORY_KIND_WEIGHTS = 1,
  CUTEAFD_MEMORY_KIND_EMBEDDING = 2,
  CUTEAFD_MEMORY_KIND_EXPERTS = 3,
  CUTEAFD_MEMORY_KIND_DRAFTER_WEIGHTS = 4,
  CUTEAFD_MEMORY_KIND_TABLES = 5,
  CUTEAFD_MEMORY_KIND_KV = 6,
  CUTEAFD_MEMORY_KIND_DRAFTER_KV = 7,
  CUTEAFD_MEMORY_KIND_PREFIX = 8,
  CUTEAFD_MEMORY_KIND_WORKSPACE = 9,
  CUTEAFD_MEMORY_KIND_DRAFTER_WORKSPACE = 10,
  CUTEAFD_MEMORY_KIND_TRANSPORT = 11,
  CUTEAFD_MEMORY_KIND_STAGING = 12,
  CUTEAFD_MEMORY_KIND_COUNT = 12,
} cuteafd_memory_kind_t;
