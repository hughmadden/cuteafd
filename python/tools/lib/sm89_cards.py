"""The SM89 (Ada) cards one SM89 coordinator export serves.

An SM120 export sizes launch capacity for its largest card (the 188-SM RTX
PRO 6000) and takes grids from the runtime SM count, so the 170-SM RTX 5090
runs the same build. An SM89 export does the same for the RTX 4090 (128 SMs,
24 or 48 GB) and the RTX 4090 D (114 SMs, 24 or 48 GB): capacity for 128 SMs,
and one exact plan per card for every kernel whose plan depends on the SM
count. Any other SM89 card finds no exact plan and is refused.

SM89 exports compile on b12x's offline target for this description, with
CUDA hidden (CUTEAFD_EXPORT_OFFLINE_TARGET=sm89, _pinned_sparkinfer.py), so
an sm_89 build needs no GPU and the card it happens to see shapes nothing.
"""

CARD_SMS = (114, 128)
WORKSPACE_SMS = max(CARD_SMS)
CAPABILITY = (8, 9)
# Every supported card's opt-in shared memory per block and per SM.
MAX_SHARED_MEMORY_PER_BLOCK = 101376
MAX_SHARED_MEMORY_PER_SM = 102400
