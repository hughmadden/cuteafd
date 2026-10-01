//! The GPU owner polls admitted QPs and sends results without work queues.
use super::*;
use cuteafd_transport::{
    ExpertProtocolV2ResponseHeader, LocalVerbsExpertConnection, ProtocolV2ExecutorResponseRef,
    RequestDisposition, SparkReduceMesh, SparkReduceMeshConfig,
};
use std::{
    net::TcpListener,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

struct Admission {
    stop: Arc<AtomicBool>,
}

/// A wave whose partial went out to the other ranks; answered once their
/// copies of this rank's rows arrive.
struct ReduceJob {
    connection: u64,
    partial: usize,
    tag: u32,
    rows: u32,
    header: ExpertProtocolV2ResponseHeader,
    started: Instant,
    computed_ms: f64,
}

/// The reduction mesh, formed on its own thread while the weights load.
enum Mesh {
    Off,
    Forming(mpsc::Receiver<Result<SparkReduceMesh>>),
    Ready(SparkReduceMesh),
    Failed(String),
}

impl Mesh {
    fn get(&mut self) -> Result<&mut SparkReduceMesh> {
        if let Mesh::Forming(receiver) = self {
            *self = match receiver.recv_timeout(Duration::from_secs(600)) {
                Ok(Ok(mesh)) => Mesh::Ready(mesh),
                Ok(Err(error)) => Mesh::Failed(format!("{error:#}")),
                Err(_) => Mesh::Failed("the reduction mesh never formed".into()),
            };
        }
        match self {
            Mesh::Ready(mesh) => Ok(mesh),
            Mesh::Off => anyhow::bail!("Spark reduction requested but this worker has no --reduce-rail"),
            Mesh::Failed(error) => anyhow::bail!("Spark reduction mesh unavailable: {error}"),
            Mesh::Forming(_) => unreachable!("resolved above"),
        }
    }
}
impl Drop for Admission {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

pub(super) fn run(config: NativeExpertServiceConfig, listen: &str) -> Result<()> {
    ensure!(
        config.rank < config.world && matches!(config.world, 2 | 3 | 4 | 6),
        "native rank must be below the launched Spark world"
    );
    ensure!(
        matches!(config.capacity, 1 | 16 | 80 | 256 | 1024 | 4096),
        "unsupported native capacity"
    );
    // The checkpoint fixes the process expert geometry before the native
    // library loads, so its helpers and every wire size agree with it.
    let catalog = cuteafd_loader::read_expert_catalog(&config.snapshot)?;
    let geometry = catalog.routed_experts().geometry()?;
    cuteafd_core::set_expert_geometry(geometry).map_err(|fixed| {
        anyhow::anyhow!("expert geometry is already {fixed:?}; the checkpoint needs {geometry:?}")
    })?;
    let minimum_frame = 128 + geometry.row_bytes() as usize + 40 + geometry.topk as usize * 12;
    ensure!(
        (minimum_frame..=64 * 1024 * 1024).contains(&config.max_frame_bytes),
        "invalid native frame budget"
    );
    let mut mesh = if config.reduce_rails.is_empty() {
        Mesh::Off
    } else {
        let mesh_config = SparkReduceMeshConfig {
            rank: config.rank,
            world: config.world,
            rails: config.reduce_rails.clone(),
            capacity_rows: config.capacity,
            row_bytes: geometry.row_bytes() as usize,
            timeout: Duration::from_secs(1800),
        };
        let (formed, receiver) = mpsc::sync_channel(1);
        thread::Builder::new().name("spark-reduce-mesh".into()).spawn(move || {
            let mesh = SparkReduceMesh::establish(mesh_config);
            if let Err(error) = &mesh {
                tracing::error!("Spark reduction mesh failed: {error:#}");
            }
            let _ = formed.send(mesh);
        })?;
        Mesh::Forming(receiver)
    };
    let library = unsafe { NativeLibrary::load(&config.library) }?;
    let (weights, remaining) = load_weights(&library, &catalog, &config)?;
    let mut execution = weights.execution(&library, &config, remaining)?;
    let mut exchange = HostExpertExchange::new(config.capacity)?;
    let mut row_indices = vec![0; config.capacity as usize];
    // Transport benchmarks only: answer each request with its response slot
    // as it is, without running the experts.
    let skip_compute = std::env::var("CUTEAFD_EXPERTD_SKIP_COMPUTE").is_ok_and(|v| v == "1");
    if skip_compute {
        tracing::warn!("CUTEAFD_EXPERTD_SKIP_COMPUTE=1: answering without running the experts");
    }
    // The mapped rings each accepted endpoint pins are bounded by the
    // capacity-sized two-endpoint allowance that admission already reserved and
    // proved against the device budget and actual free memory. A stale or larger
    // peer advertisement is rejected at accept instead of overcommitting.
    let ring_budget = match config.topology {
        Some(_) => Some(cuteafd_transport::RingBudget::new(spark_transport_bytes(&config)?)),
        None => None,
    };
    // Point-in-time ring counters for the main-thread memory milestones. The
    // bootstrap thread keeps its own clone; the atomic peak can be raised by a
    // concurrent admission, so these are sampled observations, not reservations.
    let ring_budget_log = ring_budget.clone();
    log_spark_memory_if_enabled(
        &library,
        &config,
        "execution and exchange allocated",
        None,
        ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
    );
    let listener = TcpListener::bind(listen)?;
    listener.set_nonblocking(true)?;
    let (admit, incoming) = mpsc::sync_channel(2);
    let stop = Arc::new(AtomicBool::new(false));
    let _guard = Admission { stop: stop.clone() };
    let max_frame_bytes = config.max_frame_bytes;
    // Resolved once here: the admission thread and the poll loop must not read
    // the process environment per connection or per poll.
    let protocol_v2_timing = cuteafd_transport::protocol_v2_timing_from_env();
    thread::Builder::new()
        .name("v41-roce-bootstrap".into())
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let admitted = match &ring_budget {
                            Some(budget) => LocalVerbsExpertConnection::accept_with_budget(
                                stream,
                                max_frame_bytes,
                                protocol_v2_timing,
                                Arc::clone(budget),
                            ),
                            None => LocalVerbsExpertConnection::accept(
                                stream,
                                max_frame_bytes,
                                protocol_v2_timing,
                            ),
                        };
                        match admitted {
                            Ok(connection) => {
                                if admit.try_send(connection).is_err() {
                                    tracing::warn!("native RoCE admission queue full or stopped");
                                }
                            }
                            Err(error) => tracing::warn!(%error, "native RoCE bootstrap failed"),
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => {
                        tracing::error!(%error, "native RoCE listener failed");
                        break;
                    }
                }
            }
        })?;
    let executor_id = match config.topology {
        Some(topology) => topology.executor_id(config.rank)?,
        None => cuteafd_transport::expert::v41_spark_executor_id(config.world, config.rank)?,
    };
    let mut connections = Vec::<(u64, LocalVerbsExpertConnection)>::with_capacity(16);
    let mut next_connection = 0u64;
    let mut jobs = Vec::<ReduceJob>::with_capacity(cuteafd_transport::SPARK_REDUCE_WAVES);
    let reducer = library.v41_compact_reducer()?;
    let reduce_stream = match mesh {
        Mesh::Off => std::ptr::null_mut(),
        _ => {
            ensure!(reducer.supports_row_shards(), "the native library has no Spark reduction kernels");
            library.cuda_stream_create()?
        }
    };
    let reduce_timing = std::env::var("CUTEAFD_SPARK_REDUCE_TIMING").is_ok_and(|v| v == "1");
    let row_bytes = geometry.row_bytes() as usize;
    // Structured startup evidence: one line per rank naming the native role and
    // logical intermediate this worker actually loaded. A captured log can then
    // be hashed and checked against the declared topology, instead of trusting a
    // hand-written value; the role comes from the same selection the loader used.
    let loaded = config.selection(config.first_layer)?;
    tracing::info!(
        rank = config.rank,
        world = config.world,
        role = loaded.role(),
        intermediate = weights.intermediate(),
        capacity = config.capacity,
        first_layer = config.first_layer,
        layers = weights.len(),
        "native local RoCE expert worker ready"
    );
    // Requests arrive back-to-back while serving, so the loop spins: it is the
    // wakeup path and this is the GPU owner thread. A quiet connection switches
    // to the endpoint's QP completion event wait (`ibv_req_notify_cq` +
    // completion-channel poll, which still busy-polls briefly) with an
    // `idle_wait` upper bound. One wait covers one endpoint, so while idle the
    // loop blocks on one connection per pass and rotates; the pass itself still
    // sweeps every connection with a non-blocking poll, which bounds pickup to
    // one wait window. Any request or admission resets the idle timer.
    let idle_spin = Duration::from_secs(30);
    let idle_wait = Duration::from_millis(100);
    let mut last_activity = Instant::now();
    let mut idle_cursor = 0usize;
    // One steady-state memory observation per admission event: after the owned
    // connection set changes, the next successful request triggers a single
    // sample. This is "first success after the admission event", not a proof
    // that the request arrived on the newly added connection.
    let mut pending_steady_log = false;
    loop {
        let mut progressed = false;
        if connections.is_empty() && jobs.is_empty() {
            connections.push((next_connection, incoming.recv().context("native RoCE admission stopped")?));
            next_connection += 1;
            progressed = true;
            pending_steady_log = true;
            log_spark_memory_if_enabled(
                &library,
                &config,
                "connection owned after accept",
                Some(connections.len()),
                ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
            );
        } else if let Ok(connection) = incoming.try_recv() {
            progressed = true;
            if connections.len() < 16 {
                connections.push((next_connection, connection));
                next_connection += 1;
                pending_steady_log = true;
                log_spark_memory_if_enabled(
                    &library,
                    &config,
                    "connection owned after accept",
                    Some(connections.len()),
                    ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
                );
            } else {
                tracing::warn!("native RoCE active connection limit reached");
            }
        }
        let waiting = !progressed
            && !connections.is_empty()
            && jobs.is_empty()
            && last_activity.elapsed() >= idle_spin;
        let wait_index = waiting.then(|| idle_cursor % connections.len());
        let mut index = 0;
        while index < connections.len() {
            let mut execution_failed = false;
            let wait = (wait_index == Some(index)).then_some(idle_wait);
            let connection_id = connections[index].0;
            let result = connections[index].1.poll(wait, |view, mapped, emit| {
                // A topology-bound worker admits only the ownership-aware
                // request contract; every other family is a protocol mismatch,
                // not a silent fallback.
                let request = match config.topology {
                    Some(topology) => BackboneRequest::parse_native_group(
                        view.frame_bytes(),
                        config.capacity,
                        topology,
                    )?,
                    None if execution.is_paired() =>
                        BackboneRequest::parse_paired(view.frame_bytes(), config.capacity)?,
                    None => BackboneRequest::parse(view.frame_bytes(), config.capacity)?,
                };
                if request.is_row_sharded() {
                    let started = Instant::now();
                    let mesh = mesh.get()?;
                    let tag = request_tag(&request);
                    ensure!(!jobs.iter().any(|job| job.tag == tag), "two reduction waves share tag {tag:#x}");
                    let header = request.row_shard_response_header(executor_id, config.world, config.rank)?;
                    let (partial, slot) = mesh.acquire()?;
                    if !skip_compute {
                        let computed = (|| {
                            let layer = (request.layer() as usize).checked_sub(config.first_layer)
                                .context("requested expert layer is not resident on this Spark")?;
                            execution.bind_layer(&weights, layer)?;
                            // SAFETY: the partial buffer is mapped, device-visible
                            // and claimed for this wave until `finish`.
                            unsafe { execution.execute_mapped_request(&request, executor_id, &mut exchange,
                                slot, Some(mapped.hidden_payload)) }
                        })();
                        match computed {
                            Ok(Some(_)) => {}
                            Ok(None) => {
                                mesh.abandon(partial);
                                anyhow::bail!("this expert backend has no device response path for Spark reduction");
                            }
                            Err(error) => {
                                mesh.abandon(partial);
                                execution_failed = true;
                                return Err(error);
                            }
                        }
                    }
                    let computed_ms = started.elapsed().as_secs_f64() * 1e3;
                    if let Err(error) = mesh.post(partial, tag, request.rows()) {
                        mesh.abandon(partial);
                        return Err(error);
                    }
                    jobs.push(ReduceJob { connection: connection_id, partial, tag, rows: request.rows(), header,
                        started, computed_ms });
                    return Ok(RequestDisposition::Deferred);
                }
                if skip_compute {
                    if let Some(slot) = mapped.response_slot {
                        let prefix = cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
                        let bytes = request.plane_bytes()?;
                        if request.permits_device_response() && slot.bytes >= prefix + bytes {
                            let output = cuteafd_ffi::CuteafdDeviceBuffer {
                                // SAFETY: the payload follows the header inside the mapped slot.
                                ptr: unsafe { slot.ptr.cast::<u8>().add(prefix) }.cast(), bytes, ..slot
                            };
                            return emit(ProtocolV2ExecutorResponseRef::Device(
                                request.response_device(executor_id, output)?))
                                .map(|()| RequestDisposition::Answered);
                        }
                    }
                }
                let layer = (request.layer() as usize).checked_sub(config.first_layer)
                    .context("requested expert layer is not resident on this Spark")?;
                execution.bind_layer(&weights, layer)?;
                if let Some(slot) = mapped.response_slot {
                    // The mapped frame's hidden rows are device-visible, so the
                    // worker copies them on its stream instead of uploading.
                    let response = unsafe { execution.execute_mapped_request(&request,
                        executor_id, &mut exchange, slot, Some(mapped.hidden_payload)) };
                    let response = match response {
                        Ok(response) => response,
                        Err(error) => { execution_failed = true; return Err(error); }
                    };
                    if let Some(response) = response {
                        return emit(ProtocolV2ExecutorResponseRef::Device(response))
                            .map(|()| RequestDisposition::Answered);
                    }
                }
                let mut emit_failed = false;
                let result = execution.execute_host_chunks(
                    &request,
                    executor_id,
                    &mut exchange,
                    &mut row_indices,
                    config.max_frame_bytes,
                    |response| {
                        let result = emit(ProtocolV2ExecutorResponseRef::Host(response));
                        emit_failed |= result.is_err();
                        result
                    },
                );
                execution_failed = result.is_err() && !emit_failed;
                result.map(|()| RequestDisposition::Answered)
            });
            match result {
                Ok(processed) => {
                    progressed |= processed;
                    if processed && pending_steady_log {
                        log_spark_memory_if_enabled(
                            &library,
                            &config,
                            "steady state sample after first post-admission request",
                            Some(connections.len()),
                            ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
                        );
                        pending_steady_log = false;
                    }
                    index += 1;
                }
                Err(error) => {
                    if execution_failed {
                        return Err(error).context("native GPU execution failed");
                    }
                    tracing::warn!(%error, "native RoCE peer removed");
                    connections.swap_remove(index);
                }
            }
        }
        if !jobs.is_empty() {
            let Mesh::Ready(mesh) = &mut mesh else { unreachable!("jobs exist only with a ready mesh") };
            mesh.poll().context("Spark reduction mesh failed")?;
            let mut index = 0;
            while index < jobs.len() {
                let job = &jobs[index];
                let Some(steps) = mesh.ready(job.partial, job.tag, job.rows)? else {
                    index += 1;
                    continue;
                };
                let arrived_ms = job.started.elapsed().as_secs_f64() * 1e3;
                let owner = connections.iter().position(|(id, _)| *id == job.connection);
                let mut answered = Ok(());
                if let Some(position) = owner {
                    let connection = &mut connections[position].1;
                    answered = (|| -> Result<()> {
                        let slot = connection.deferred_response_slot()?;
                        let base = slot.ptr as usize + cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
                        ensure!(slot.bytes >= cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
                            + job.header.output_payload_bytes as usize, "reduced rows exceed the response slot");
                        for step in &steps {
                            let output = (base + step.row_offset as usize * row_bytes) as *mut u16;
                            // SAFETY: the slices are this wave's received and own
                            // rows (mapped, device-visible); the output range lies
                            // inside the reserved response slot (checked above).
                            unsafe {
                                reducer.reduce_rank_slices(step.slices, config.world as u32, output,
                                    step.rows as u64 * (row_bytes / 2) as u64, reduce_stream)?;
                            }
                        }
                        // SAFETY: the stream was created above and is live.
                        unsafe { library.cuda_stream_synchronize(reduce_stream)? };
                        Ok(())
                    })();
                }
                mesh.finish(job.partial, job.tag)?;
                let job = jobs.swap_remove(index);
                if let (Some(position), Ok(())) = (owner, &answered) {
                    answered = connections[position].1.complete_deferred(job.header.clone());
                }
                if reduce_timing {
                    eprintln!("spark_reduce wave={:#x} rows={} share={} compute_ms={:.3} exchange_ms={:.3} total_ms={:.3}",
                        job.tag, job.rows, job.header.row_count, job.computed_ms, arrived_ms - job.computed_ms,
                        job.started.elapsed().as_secs_f64() * 1e3);
                }
                if let (Some(position), Err(error)) = (owner, answered) {
                    tracing::warn!(%error, "native RoCE peer removed while answering a reduced wave");
                    connections.swap_remove(position);
                }
                progressed = true;
            }
        }
        if progressed {
            last_activity = Instant::now();
        }
        if waiting {
            idle_cursor = idle_cursor.wrapping_add(1);
        } else {
            std::hint::spin_loop();
        }
    }
}

/// The wave tag peers attach to their slices: the request id the coordinator
/// sent every rank (unique among waves in flight).
fn request_tag(request: &BackboneRequest<'_>) -> u32 {
    request.request_id() as u32
}
