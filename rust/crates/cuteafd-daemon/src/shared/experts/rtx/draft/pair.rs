//! Capture-compatible draft expert fork/join on the caller-selected owner.
//!
//! The owner quantizes the rows once; both GPUs run their expert halves; the
//! owner reduces. Rows cross between the GPUs in one of two ways:
//! - pull (default): the peer's SMs read the owner's encoded inputs after a
//!   cross-device event, and the owner's SMs read the peer's routes after
//!   another;
//! - push ([`Pair::enable_push`]): the GPU that produced the rows writes them
//!   into the other GPU's buffer on its own stream and publishes a device
//!   flag, which the consumer's stream waits on.
//!
//! Either way the events still fork the peer stream from the owner's and join
//! it back, so a capture of the owner stream takes the peer branch with it.
//! Flag sequences live on the device (positional), so graph replays and eager
//! runs pair each wait with its own push.
use super::{
    rank::{Wave, Weights},
    Layout,
};
use crate::shared::memory::device::{Allocation, Device, Event, Stream};
use crate::shared::peer_split::PeerExchange;
use anyhow::{ensure, Result};
use cuteafd_ffi::{
    peer_exchange::PeerPlane, CuteafdDeviceBuffer, NativeLibrary, V41ExpertInputQuantizer,
    V41PeerCopy, V41RouteReducer,
};
use std::{cell::Cell, ffi::c_void, rc::Rc};

/// Ranks of a pushing pair's flag exchange.
const OWNER: usize = 0;
const PEER: usize = 1;
/// Its flags: the owner's encoded inputs, then the peer's routes.
const INPUT: usize = 0;
const ROUTES: usize = 1;
const FLAGS: usize = 2;

/// Bytes of an input plane pushed in whole 16-byte units (route ids and
/// weights are 12 bytes a row); input allocations are whole units too.
pub(crate) fn padded(bytes: usize) -> usize { bytes.next_multiple_of(16) }

pub(crate) struct Pair<'a> {
    // External root stream and its graphs must be drained/destroyed by caller
    // first. Our peer stream drains before any of its referenced allocations.
    peer: Stream<'a>,
    /// `enable_push`: flags of the pushed exchange (ranks [`OWNER`], [`PEER`]).
    push: Option<PeerExchange<'a>>,
    /// Set from a pushing enqueue's first push until its last wait is
    /// queued: still set means a failed enqueue left a push without its wait,
    /// the positional flags are out of step, and the pair refuses more work.
    unmatched: Cell<bool>,
    ready: Event<'a>,
    done: Event<'a>,
    ranks: [Wave<'a>; 2],
    wire: Allocation<'a>,
    peer_inputs: [Allocation<'a>; 3],
    peer_output: Allocation<'a>,
    output: Allocation<'a>,
    copies: [V41PeerCopy<'a>; 2],
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
        let inputs = layout.input_bytes().map(|width| padded(rows * width));
        // Flag control words on both GPUs, whether or not the pair pushes.
        let mut bytes = [scratch + PeerExchange::flag_bytes(FLAGS); 2];
        bytes[1 - layout.owner] += inputs.iter().sum::<usize>();
        bytes[layout.owner] +=
            inputs[0] + rows * (layout.route_bytes() + layout.geometry.row_bytes() as usize);
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
        let copies = [
            peer.run(|| owner.library.v41_peer_copy())?,
            owner.run(|| owner.library.v41_peer_copy())?,
        ];
        let quantizer = owner.run(|| owner.library.v41_expert_input_quantizer())?;
        let reducer = owner.library.v41_route_reducer()?;
        let [left, right] = weights;
        let ranks = [Wave::new(left, capacity)?, Wave::new(right, capacity)?];
        let rows = capacity as usize;
        let [wire, ids, weights] = layout.input_bytes().map(|width| padded(rows * width));
        Ok(Self {
            peer: Stream::new(peer)?,
            push: None,
            unmatched: Cell::new(false),
            ready: Event::new(owner)?,
            done: Event::new(peer)?,
            ranks,
            wire: Allocation::new(owner, wire)?,
            peer_inputs: [
                Allocation::new(peer, wire)?,
                Allocation::new(peer, ids)?,
                Allocation::new(peer, weights)?,
            ],
            peer_output: Allocation::new(owner, rows * layout.route_bytes())?,
            output: Allocation::new(owner, rows * layout.geometry.row_bytes() as usize)?,
            copies,
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
    /// Push instead of pull from now on: SM pushes up to `decode_lane_bytes`
    /// a transfer, the sender's copy engine plus a zero-byte flag above.
    /// Reuse needs no slots: the next push into either GPU's buffers follows
    /// that buffer's last reader through the routes flag, the reduction and
    /// stream order (`v41_peer_push::schedule_tests` checks a draft round).
    pub fn enable_push(&mut self, decode_lane_bytes: usize) -> Result<()> {
        let devices = [self.owner.id, self.peer.device.id];
        self.push = Some(PeerExchange::new_flags(self.owner.library, devices, FLAGS, decode_lane_bytes)?);
        Ok(())
    }
    pub fn pushes(&self) -> bool {
        self.push.is_some()
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
        if self.push.is_some() {
            return unsafe { self.enqueue_pushed(stage, rows, inputs, shared, stream) };
        }
        self.owner.run(|| unsafe {
            self.quantizer
                .launch(inputs[0], self.wire.buffer, rows, stream)?;
            library.cuda_event_record(self.ready.raw, stream)
        })?;
        let encoded = [self.wire.buffer, inputs[1], inputs[2]];
        let peer_result = self.peer.device.run(|| unsafe {
            library.cuda_stream_wait_event(self.peer.raw, self.ready.raw)?;
            for ((destination, source), width) in self
                .peer_inputs
                .iter()
                .zip(encoded)
                .zip(self.layout.input_bytes())
            {
                self.copies[0].launch(
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
            library.cuda_event_record(self.done.raw, self.peer.raw)?;
            Ok(output)
        })?;
        self.owner.run(|| unsafe {
            let local = self.ranks[self.layout.owner].enqueue(stage, rows, encoded, stream)?;
            library.cuda_stream_wait_event(stream, self.done.raw)?;
            self.copies[1].launch(
                self.peer_output.buffer,
                peer_result,
                rows as usize * self.layout.route_bytes(),
                stream,
            )?;
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
    /// [`Self::enqueue`] with pushes. Every push is queued before its wait:
    /// - owner: quantize, fork (`ready`), push the encoded inputs (flag
    ///   [`INPUT`]), its half;
    /// - peer: fork, wait for [`INPUT`], its half, push its routes (flag
    ///   [`ROUTES`]), join (`done`);
    /// - owner: wait for [`ROUTES`], reduce, then the join, which only closes
    ///   the capture topology: the flag already ordered the routes.
    ///
    /// # Safety
    /// As [`Self::enqueue`]; route ids and weights hold whole 16-byte units
    /// past `rows` (the plane rounds up to them).
    unsafe fn enqueue_pushed(
        &mut self,
        stage: usize,
        rows: u32,
        inputs: [CuteafdDeviceBuffer; 3],
        shared: Option<CuteafdDeviceBuffer>,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(!self.unmatched.get(),
            "draft TP2 flags are out of step after a failed enqueue; rebuild the pair");
        let library = self.owner.library;
        let encoded = [self.wire.buffer, inputs[1], inputs[2]];
        let planes = encoded.iter().zip(&self.peer_inputs).zip(self.layout.input_bytes())
            .map(|((source, destination), width)| {
                let bytes = padded(rows as usize * width);
                ensure!(bytes <= source.bytes && bytes <= destination.buffer.bytes,
                    "draft TP2 pushed input plane exceeds its allocation");
                Ok(PeerPlane { destination: destination.buffer.ptr, source: source.ptr.cast_const(),
                    rows: 1, row_bytes: bytes as u64, destination_pitch: bytes as u64, source_pitch: bytes as u64 })
            })
            .collect::<Result<Vec<_>>>()?;
        let flags = self.push.as_ref().expect("pushing pair");
        self.owner.run(|| unsafe {
            self.quantizer.launch(inputs[0], self.wire.buffer, rows, stream)?;
            library.cuda_event_record(self.ready.raw, stream)
        })?;
        self.unmatched.set(true);
        let local = self.owner.run(|| unsafe {
            flags.push_planes_on(OWNER, stream, INPUT, &planes)?;
            self.ranks[self.layout.owner].enqueue(stage, rows, encoded, stream)
        })?;
        let route_bytes = rows as usize * self.layout.route_bytes();
        self.peer.device.run(|| unsafe {
            library.cuda_stream_wait_event(self.peer.raw, self.ready.raw)?;
            flags.wait_on(PEER, self.peer.raw, INPUT)?;
            let routes = self.ranks[1 - self.layout.owner].enqueue(
                stage,
                rows,
                self.peer_inputs.each_ref().map(|v| v.buffer),
                self.peer.raw,
            )?;
            flags.push_to_on(PEER, self.peer.raw, ROUTES, routes.ptr.cast_const(),
                self.peer_output.buffer.ptr, route_bytes)?;
            library.cuda_event_record(self.done.raw, self.peer.raw)
        })?;
        self.owner.run(|| unsafe {
            flags.wait_on(OWNER, stream, ROUTES)?;
            self.unmatched.set(false);
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
            )?;
            library.cuda_stream_wait_event(stream, self.done.raw)
        })
    }
    /// Used after ending a failed capture or after root completion on error.
    pub fn drain_peer(&self) -> Result<()> {
        self.peer.drain()
    }
}
