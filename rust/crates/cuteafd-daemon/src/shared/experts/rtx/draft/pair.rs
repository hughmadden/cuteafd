//! Capture-compatible draft expert fork/join on the caller-selected owner.
use super::{
    rank::{Wave, Weights},
    Layout,
};
use crate::shared::memory::device::{Allocation, Device, Event, Stream};
use anyhow::{ensure, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41ExpertInputQuantizer, V41PeerCopy, V41RouteReducer,
};
use std::{ffi::c_void, rc::Rc};
use crate::shared::memory::Tag;

pub(crate) struct Pair<'a> {
    // External root stream and its graphs must be drained/destroyed by caller
    // first. Our peer stream drains before any of its referenced allocations.
    peer: Rc<Stream<'a>>,
    ready: Event<'a>,
    done: Event<'a>,
    ranks: [Wave<'a>; 2],
    wire: Allocation<'a>,
    peer_inputs: [Allocation<'a>; 3],
    peer_output: Allocation<'a>,
    output: Allocation<'a>,
    copy: V41PeerCopy<'a>,
    quantizer: V41ExpertInputQuantizer<'a>,
    reducer: V41RouteReducer<'a>,
    owner: Device<'a>,
    capacity: u32,
    layout: Layout,
}
impl<'a> Pair<'a> {
    pub fn device_bytes(
        library: &NativeLibrary,
        capacity: u32,
        layout: Layout,
    ) -> Result<[usize; 2]> {
        layout.validate()?;
        let scratch = Wave::device_bytes(library, capacity, layout)?;
        let rows = capacity as usize;
        let widths = layout.input_bytes();
        let mut bytes = [scratch; 2];
        bytes[1 - layout.owner] += rows * widths.iter().sum::<usize>();
        bytes[layout.owner] +=
            rows * (widths[0] + layout.route_bytes() + layout.geometry.row_bytes() as usize);
        Ok(bytes)
    }
    pub fn new(weights: [Rc<Weights<'a>>; 2], capacity: u32, layout: Layout) -> Result<Self> {
        layout.validate()?;
        let devices = weights.each_ref().map(|w| w.device);
        ensure!(
            devices[0].id != devices[1].id
                && std::ptr::eq(devices[0].library, devices[1].library)
                && weights.iter().all(|w| w.layout == layout),
            "invalid draft TP2 weights/layout"
        );
        let owner = devices[layout.owner];
        let peer = devices[1 - layout.owner];
        for (rank, device) in devices.iter().enumerate() {
            device.run(|| device.library.cuda_enable_peer(devices[1 - rank].id))?;
        }
        let copy = peer.run(|| owner.library.v41_peer_copy())?;
        let quantizer = owner.run(|| owner.library.v41_expert_input_quantizer())?;
        let reducer = owner.library.v41_route_reducer()?;
        let [left, right] = weights;
        let ranks = [Wave::new(left, capacity)?, Wave::new(right, capacity)?];
        let rows = capacity as usize;
        let widths = layout.input_bytes();
        Ok(Self {
            peer: Rc::new(Stream::new(peer)?),
            ready: Event::new(owner)?,
            done: Event::new(peer)?,
            ranks,
            wire: Allocation::new(owner, rows * widths[0], Tag::drafter_workspace("experts/draft-pair"))?,
            peer_inputs: [
                Allocation::new(peer, rows * widths[0], Tag::transport("experts/draft-pair"))?,
                Allocation::new(peer, rows * widths[1], Tag::transport("experts/draft-pair"))?,
                Allocation::new(peer, rows * widths[2], Tag::transport("experts/draft-pair"))?,
            ],
            peer_output: Allocation::new(owner, rows * layout.route_bytes(), Tag::transport("experts/draft-pair"))?,
            output: Allocation::new(owner, rows * layout.geometry.row_bytes() as usize, Tag::drafter_workspace("experts/draft-pair"))?,
            copy,
            quantizer,
            reducer,
            owner,
            capacity,
            layout,
        })
    }
    pub fn output(&self) -> CuteafdDeviceBuffer {
        self.output.buffer
    }
    /// Quantize once on the owner, fork the peer, then join its FP32 route contributions
    /// before ordered top-3 reduction. No allocations or host waits here.
    ///
    /// # Safety
    /// Caller owns the owner stream and all inputs/shared through completion and
    /// graph lifetime. Prior use of this pair must precede this enqueue. On any
    /// failure, end/abort capture as appropriate and drain both streams before
    /// freeing or reusing any inputs. Graph destruction precedes dropping Pair.
    pub unsafe fn enqueue(
        &mut self,
        stage: usize,
        rows: u32,
        inputs: [CuteafdDeviceBuffer; 3],
        shared: Option<CuteafdDeviceBuffer>,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            stage < self.layout.stages && rows > 0 && rows <= self.capacity,
            "invalid draft TP2 stage/rows"
        );
        for (buffer, width) in inputs.iter().zip([
            self.layout.geometry.row_bytes() as usize,
            self.layout.geometry.topk as usize * 4,
            self.layout.geometry.topk as usize * 4,
        ]) {
            ensure!(
                buffer.device_id == self.owner.id && buffer.bytes >= rows as usize * width,
                "draft TP2 root input device or size differs"
            );
        }
        if let Some(buffer) = shared {
            ensure!(
                buffer.device_id == self.owner.id
                    && buffer.bytes >= rows as usize * self.layout.geometry.row_bytes() as usize,
                "draft TP2 shared output device or size differs"
            );
        }
        let library = self.owner.library;
        self.owner.run(|| unsafe {
            self.quantizer
                .launch(inputs[0], self.wire.buffer, rows, stream)?;
            library.cuda_event_record(self.ready.raw, stream)
        })?;
        let encoded = [self.wire.buffer, inputs[1], inputs[2]];
        self.peer.device.run(|| unsafe {
            library.cuda_stream_wait_event(self.peer.raw, self.ready.raw)?;
            for ((destination, source), width) in self
                .peer_inputs
                .iter()
                .zip(encoded)
                .zip(self.layout.input_bytes())
            {
                self.copy.launch(
                    destination.buffer,
                    source,
                    rows as usize * width,
                    self.peer.raw,
                )?;
            }
            let output = self.ranks[1 - self.layout.owner].enqueue(
                stage,
                rows,
                self.peer_inputs.each_ref().map(|v| v.buffer),
                self.peer.raw,
            )?;
            // Push completed FP32 routes from their producer; the owner's join
            // covers the peer write as well as its expert kernels.
            self.copy.launch(
                self.peer_output.buffer,
                output,
                rows as usize * self.layout.route_bytes(),
                self.peer.raw,
            )?;
            library.cuda_event_record(self.done.raw, self.peer.raw)
        })?;
        self.owner.run(|| unsafe {
            let local = self.ranks[self.layout.owner].enqueue(stage, rows, encoded, stream)?;
            library.cuda_stream_wait_event(stream, self.done.raw)?;
            let mut planes = [std::ptr::null(); 4];
            planes[self.layout.owner] = local.ptr.cast();
            planes[1 - self.layout.owner] = self.peer_output.buffer.ptr.cast();
            self.reducer.launch(
                planes,
                shared.map_or(std::ptr::null(), |v| v.ptr.cast()),
                self.output.buffer.ptr.cast(),
                rows,
                2,
                self.layout.geometry.topk,
                stream,
            )
        })
    }
    /// Keep the peer alive in submission guards, including before a join exists.
    pub fn peer_stream(&self) -> Rc<Stream<'a>> {
        Rc::clone(&self.peer)
    }
    /// Used after ending a failed capture or after root completion on error.
    pub fn drain_peer(&self) -> Result<()> {
        self.peer.drain()
    }
}
