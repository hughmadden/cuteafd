#!/usr/bin/env python3
"""Official-weight CSA2 compressor projection/pooling/normalization qualification."""
import argparse
import ast
import ctypes as C
import hashlib
import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace
import torch


def row_invariance(args):
    """Bitwise compare official-weight native projections across live widths."""
    widths=(1,2,3,4,5,6,15,16,17,38,64,80)
    torch.cuda.set_device(args.device);torch.manual_seed(410233)
    lib=C.CDLL(str(args.native_lib));P=C.c_void_p;I=C.c_int32
    def bind(name,types):
        fn=getattr(lib,name);fn.argtypes=types;fn.restype=I;return fn
    create=bind('cuteafd_v41_compressor_create',[P,C.c_uint64,C.POINTER(P)])
    destroy=bind('cuteafd_v41_compressor_destroy',[P])
    project=bind('cuteafd_v41_compressor_project',[P,P,P,P,I,I,P])
    key=bind('cuteafd_v41_index_key_project',[P,P,P,P,I,P])
    heads=bind('cuteafd_v41_index_weights_project',[P,P,P,P,I,P])
    def check(status):assert status==0,status
    index=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    hashes={};results=[];unaligned_rejections=0
    def load(name):
        with (args.snapshot/index[name]).open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0];entry=json.loads(f.read(size))[name]
            assert entry['dtype']=='BF16'
            a,b=entry['data_offsets'];f.seek(8+size+a);data=f.read(b-a);assert len(data)==b-a
        hashes[name]=hashlib.sha256(data).hexdigest()
        return torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(entry['shape']).cuda()
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    workspaces=[torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8) for _ in streams]
    handles=[P(),P()]
    for workspace,handle in zip(workspaces,handles):check(create(workspace.data_ptr(),workspace.numel(),C.byref(handle)))
    try:
        for layer in (2,8,14,20):
            ratio=1 if layer==20 else 2
            cases=[(f'layers.{layer}.attn.compressor.{s}.weight',project,ratio)
                   for s in (('wkv',) if ratio==1 else ('wkv','wgate'))]
            cases.extend([(f'layers.{layer}.attn.indexer.wk.weight',key,1),
                          (f'layers.{layer}.attn.indexer.weights_proj.weight',heads,1)])
            for name,fn,ratio in cases:
                weight=load(name);n,k=weight.shape
                x=(torch.randn(max(widths),k,device='cuda')*.125).bfloat16()
                dtype=torch.float32 if ratio==2 else torch.bfloat16
                bits=torch.int32 if ratio==2 else torch.int16
                torch.cuda.synchronize();serial=None
                for lane,(stream,handle) in enumerate(zip(streams,handles)):
                    with torch.cuda.stream(stream),torch.no_grad():
                        def run(first,rows):
                            out=torch.empty(rows,n,device='cuda',dtype=dtype)
                            argv=[handle,x[first:first+rows].data_ptr(),weight.data_ptr(),out.data_ptr(),rows]
                            if fn is project:argv.append(ratio)
                            check(fn(*argv,stream.cuda_stream));return out
                        # Exercise the native ABI guard, not only the Rust preflight.
                        rejected_output=torch.empty(1,n,device='cuda',dtype=dtype)
                        pointers=[x.data_ptr(),weight.data_ptr(),rejected_output.data_ptr()]
                        for operand in range(3):
                            unaligned=pointers.copy();unaligned[operand]+=2
                            argv=[handle,*unaligned,1]
                            if fn is project:argv.append(ratio)
                            assert fn(*argv,stream.cuda_stream)==1, (name,operand)
                            unaligned_rejections+=1
                        current=torch.cat([run(i,1) for i in range(max(widths))]);stream.synchronize()
                        if serial is None:serial=current
                        assert torch.equal(current.view(bits),serial.view(bits)),name
                        for rows in widths:
                            got=run(0,rows);stream.synchronize()
                            counts=(got.view(bits)!=serial[:rows].view(bits)).sum(1).tolist()
                            record={'tensor':name,'shape':[n,k],'dtype':str(dtype),'lane':lane,'rows':rows,
                                    'different_elements':sum(counts),'different_by_row':counts,
                                    'max_abs_error':(got.float()-serial[:rows].float()).abs().max().item()}
                            results.append(record)
                            print('PASS' if not sum(counts) else 'FAIL',json.dumps(record),flush=True)
                del weight,x,serial
    finally:
        for stream,handle in zip(streams,handles):stream.synchronize();check(destroy(handle))
    assert len(hashes)==15 and len(results)==360 and unaligned_rejections==90
    failed=sum(r['different_elements']!=0 for r in results)
    args.output.write_text(json.dumps({'scope':'Official-weight compressor/index projections; synthetic inputs; no serving',
        'device':args.device,'name':torch.cuda.get_device_name(args.device),'widths':widths,
        'checkpoint_payload_sha256':hashes,'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'results':results,'failed':failed,'unaligned_rejections':unaligned_rejections},indent=2)+'\n')
    print(f'SUMMARY {len(results)-failed} passed; {failed} failed; {len(results)} comparisons',flush=True)
    return 1 if failed else 0


def projection_timing(args):
    """Compare warmed direct/captured native projection costs on identical inputs."""
    assert args.baseline_native_lib is not None, '--baseline-native-lib required'
    widths=(1,6,38,2048,4096)
    torch.cuda.set_device(args.device);torch.manual_seed(410233)
    P=C.c_void_p;I=C.c_int32
    cases=[('layers.2.attn.compressor.wkv.weight','cuteafd_v41_compressor_project',2),
           ('layers.20.attn.compressor.wkv.weight','cuteafd_v41_compressor_project',1),
           ('layers.2.attn.indexer.wk.weight','cuteafd_v41_index_key_project',1),
           ('layers.2.attn.indexer.weights_proj.weight','cuteafd_v41_index_weights_project',1)]
    index=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    libraries=[];results=[];hashes={}
    stream=torch.cuda.Stream()
    for label,path in [('heuristic',args.baseline_native_lib),('fixed',args.native_lib)]:
        lib=C.CDLL(str(path));workspace=torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8);handle=P()
        create=lib.cuteafd_v41_compressor_create;create.argtypes=[P,C.c_uint64,C.POINTER(P)];create.restype=I
        destroy=lib.cuteafd_v41_compressor_destroy;destroy.argtypes=[P];destroy.restype=I
        started=time.perf_counter();assert create(workspace.data_ptr(),workspace.numel(),C.byref(handle))==0
        print('CREATE',label,(time.perf_counter()-started)*1e3,'ms',flush=True)
        libraries.append((label,lib,workspace,handle,destroy))
    try:
        for name,symbol,ratio in cases:
            with (args.snapshot/index[name]).open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0];entry=json.loads(f.read(size))[name]
                a,b=entry['data_offsets'];f.seek(8+size+a);data=f.read(b-a);assert len(data)==b-a
            hashes[name]=hashlib.sha256(data).hexdigest()
            weight=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(entry['shape']).cuda()
            n,k=weight.shape;x=(torch.randn(max(widths),k,device='cuda')*.125).bfloat16()
            torch.cuda.synchronize()
            for rows in widths:
                for label,lib,workspace,handle,destroy in libraries:
                    fn=getattr(lib,symbol);fn.argtypes=[P,P,P,P,I]+([I] if 'compressor_project' in symbol else [])+[P];fn.restype=I
                    out=torch.empty(rows,n,device='cuda',dtype=torch.float32 if ratio==2 else torch.bfloat16)
                    argv=[handle,x.data_ptr(),weight.data_ptr(),out.data_ptr(),rows]+([ratio] if 'compressor_project' in symbol else [])+[stream.cuda_stream]
                    def launch():assert fn(*argv)==0
                    with torch.cuda.stream(stream):
                        for _ in range(20):launch()
                        stream.synchronize();graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):
                            for _ in range(20):launch()
                        for mode,run,operations in [('direct',launch,1),('graph',graph.replay,20)]:
                            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            start.record(stream);cpu_start=time.perf_counter()
                            for _ in range(100):run()
                            cpu_us=(time.perf_counter()-cpu_start)*1e6/(100*operations)
                            end.record(stream);end.synchronize()
                            record={'tensor':name,'shape':[n,k],'rows':rows,'arm':label,'mode':mode,
                                    'device_us':start.elapsed_time(end)*1000/(100*operations),'host_us':cpu_us}
                            results.append(record);print(json.dumps(record),flush=True)
                        graph.reset()
    finally:
        stream.synchronize()
        for label,lib,workspace,handle,destroy in libraries:assert destroy(handle)==0
    assert len(results)==80
    args.output.write_text(json.dumps({'scope':'Warmed direct/captured projection costs; synthetic activations; no serving',
        'device':torch.cuda.get_device_name(args.device),'results':results,'checkpoint_payload_sha256':hashes,
        'native_library_sha256':{label:hashlib.sha256(path.read_bytes()).hexdigest()
            for label,path in [('heuristic',args.baseline_native_lib),('fixed',args.native_lib)]}},indent=2)+'\n')
    print('SUMMARY 80 timings completed',flush=True)
    return 0


def projection_algorithms(args):
    """Screen unsplit Lt choices without changing a serving handle or ABI."""
    torch.cuda.set_device(args.device);torch.manual_seed(410233)
    P=C.c_void_p;I=C.c_int;U=C.c_uint32;Z=C.c_size_t
    class Algorithm(C.Structure):_fields_=[('data',C.c_uint64*8)]
    class Heuristic(C.Structure):
        _fields_=[('algo',Algorithm),('workspaceSize',Z),('state',I),('wavesCount',C.c_float),('reserved',I*4)]
    lib=C.CDLL('libcublasLt.so.13')
    def bind(name,types):
        f=getattr(lib,name);f.argtypes=types;f.restype=I;return f
    create=bind('cublasLtCreate',[C.POINTER(P)]);destroy=bind('cublasLtDestroy',[P])
    desc_create=bind('cublasLtMatmulDescCreate',[C.POINTER(P),I,I])
    desc_set=bind('cublasLtMatmulDescSetAttribute',[P,I,P,Z])
    desc_destroy=bind('cublasLtMatmulDescDestroy',[P])
    layout_create=bind('cublasLtMatrixLayoutCreate',[C.POINTER(P),I,C.c_uint64,C.c_uint64,C.c_int64])
    layout_set=bind('cublasLtMatrixLayoutSetAttribute',[P,I,P,Z]);layout_destroy=bind('cublasLtMatrixLayoutDestroy',[P])
    pref_create=bind('cublasLtMatmulPreferenceCreate',[C.POINTER(P)])
    pref_set=bind('cublasLtMatmulPreferenceSetAttribute',[P,I,P,Z]);pref_destroy=bind('cublasLtMatmulPreferenceDestroy',[P])
    heuristic=bind('cublasLtMatmulAlgoGetHeuristic',[P,P,P,P,P,P,P,I,C.POINTER(Heuristic),C.POINTER(I)])
    algo_set=bind('cublasLtMatmulAlgoConfigSetAttribute',[C.POINTER(Algorithm),I,P,Z])
    algo_get=bind('cublasLtMatmulAlgoConfigGetAttribute',[C.POINTER(Algorithm),I,P,Z,C.POINTER(Z)])
    algo_check=bind('cublasLtMatmulAlgoCheck',[P,P,P,P,P,P,C.POINTER(Algorithm),C.POINTER(Heuristic)])
    matmul=bind('cublasLtMatmul',[P,P,P,P,P,P,P,P,P,P,P,P,C.POINTER(Algorithm),P,Z,P])
    def check(s):assert s==0,s
    handle=P();check(create(C.byref(handle)))
    workspace=torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8)
    stream=torch.cuda.Stream();widths=(1,2,3,4,5,6,7,15,16,17,38,64,80)
    timing_widths=(1,6,38,2048,4096);records=[]
    for n,k,dtype in [(512,5120,0),(512,5120,14),(128,512,14),(32,5120,14)]:
        operation=P();a=P();b=P();out_layout=P();preference=P()
        check(desc_create(C.byref(operation),68,0));transpose=I(1)
        check(desc_set(operation,3,C.byref(transpose),C.sizeof(transpose)))
        check(layout_create(C.byref(a),14,k,n,k));check(layout_create(C.byref(b),14,k,16,k))
        check(layout_create(C.byref(out_layout),dtype,n,16,n));check(pref_create(C.byref(preference)))
        size=Z(workspace.numel());check(pref_set(preference,1,C.byref(size),C.sizeof(size)))
        alignment=U(args.projection_alignment)
        for attribute in (5,6,7,8):check(pref_set(preference,attribute,C.byref(alignment),C.sizeof(alignment)))
        def rows_set(rows):
            value=C.c_uint64(rows)
            for layout in (b,out_layout):check(layout_set(layout,3,C.byref(value),C.sizeof(value)))
        candidates={}
        for canonical in (1,6,16,38,80,4096):
            rows_set(canonical);found=(Heuristic*32)();count=I()
            check(heuristic(handle,operation,a,b,out_layout,out_layout,preference,32,found,C.byref(count)))
            for rank,h in enumerate(found[:count.value]):
                if h.state:continue
                algorithm=Algorithm.from_buffer_copy(h.algo);zero=U(0)
                if algo_set(C.byref(algorithm),2,C.byref(zero),4) or algo_set(C.byref(algorithm),3,C.byref(zero),4):continue
                candidates.setdefault(bytes(algorithm),(canonical,rank,algorithm))
        x=(torch.randn(4096,k,device='cuda')*.125).bfloat16();weight=(torch.randn(n,k,device='cuda')*.125).bfloat16()
        out=torch.empty(4096,n,device='cuda',dtype=torch.float32 if dtype==0 else torch.bfloat16)
        bits=torch.int32 if dtype==0 else torch.int16
        torch.cuda.synchronize();alpha=C.c_float(1);beta=C.c_float(0)
        for canonical,rank,algorithm in candidates.values():
            supported=True
            for rows in (*widths,128,256,1024,2048,4096):
                rows_set(rows);h=Heuristic()
                if algo_check(handle,operation,a,b,out_layout,out_layout,C.byref(algorithm),C.byref(h)) or h.state or h.workspaceSize>workspace.numel():supported=False;break
            if not supported:continue
            def launch():check(matmul(handle,operation,C.byref(alpha),weight.data_ptr(),a,x.data_ptr(),b,
                C.byref(beta),out.data_ptr(),out_layout,out.data_ptr(),out_layout,C.byref(algorithm),
                workspace.data_ptr(),workspace.numel(),stream.cuda_stream))
            with torch.cuda.stream(stream):
                rows_set(1)
                # Offset input pointer for the M1 reference; layouts are unchanged.
                for row in range(max(widths)):
                    check(matmul(handle,operation,C.byref(alpha),weight.data_ptr(),a,x[row].data_ptr(),b,
                        C.byref(beta),out[row].data_ptr(),out_layout,out[row].data_ptr(),out_layout,
                        C.byref(algorithm),workspace.data_ptr(),workspace.numel(),stream.cuda_stream))
                reference=out[:max(widths)].clone();stream.synchronize();different={}
                for rows in widths:
                    rows_set(rows);launch();stream.synchronize()
                    different[str(rows)]=(out[:rows].view(bits)!=reference[:rows].view(bits)).sum().item()
                attributes={}
                for attribute in range(10):
                    value=I();written=Z()
                    if not algo_get(C.byref(algorithm),attribute,C.byref(value),4,C.byref(written)):attributes[str(attribute)]=value.value
                record={'shape':[n,k],'dtype':dtype,'canonical':canonical,'rank':rank,'attributes':attributes,
                        'algorithm_bytes':bytes(algorithm).hex(),'different_elements':different,'graph_us':{}}
                if not any(different.values()):
                    for rows in timing_widths:
                        rows_set(rows)
                        for _ in range(10):launch()
                        stream.synchronize();graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):
                            for _ in range(20):launch()
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        for _ in range(30):graph.replay()
                        end.record(stream);end.synchronize()
                        record['graph_us'][str(rows)]=start.elapsed_time(end)*1000/600
                        graph.reset()
                records.append(record);print(json.dumps(record),flush=True)
        for layout in (a,b,out_layout):check(layout_destroy(layout))
        check(pref_destroy(preference));check(desc_destroy(operation))
    stream.synchronize();check(destroy(handle))
    args.output.write_text(json.dumps({'scope':'Synthetic BF16 inputs/weights; unsplit Lt algorithm screen; not official-weight gate',
        'device':torch.cuda.get_device_name(args.device),'widths':widths,'records':records},indent=2)+'\n')
    print('SUMMARY',len(records),'supported candidates;',sum(not any(r['different_elements'].values()) for r in records),'row-invariant',flush=True)
    return 0


def head_row_invariance(args):
    """Qualify the serving W8A16 head's GROUPS1/8 variants and both TP2 shards."""
    widths=(1,6,38)
    torch.cuda.set_device(args.device);torch.manual_seed(410233)
    lib=C.CDLL(str(args.native_lib));P=C.c_void_p;I=C.c_int32
    pack=lib.cuteafd_fp8_w8a16_pack
    pack.argtypes=[P,P,P,I,I,I,P];pack.restype=I
    workspace_bytes=lib.cuteafd_fp8_w8a16_workspace
    workspace_bytes.argtypes=[I,I,I];workspace_bytes.restype=C.c_size_t
    linear=lib.cuteafd_fp8_w8a16_linear
    linear.argtypes=[P,P,P,P,I,I,I,I,P,C.c_size_t,P];linear.restype=I
    index=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    with (args.snapshot/index['head.weight']).open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];entry=json.loads(f.read(size))['head.weight']
        assert entry['dtype']=='BF16' and entry['shape']==[129280,5120]
        a,b=entry['data_offsets'];f.seek(8+size+a);data=f.read(b-a);assert len(data)==b-a
    payload_hash=hashlib.sha256(data).hexdigest()
    weight=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(entry['shape']).cuda()
    del data
    x=(torch.randn(max(widths),5120,device='cuda')*.125).bfloat16()
    streams=[torch.cuda.Stream(),torch.cuda.Stream()];results=[]
    torch.cuda.synchronize()
    for first,n in ((0,129280),(0,64640),(64640,64640)):
        packed=torch.empty(n*5120,device='cuda',dtype=torch.uint8)
        scales=torch.empty(n,40,device='cuda',dtype=torch.float32)
        # Serving packs BF16 head weights with the per-block Best scale rule (2).
        assert pack(weight[first:first+n].data_ptr(),packed.data_ptr(),scales.data_ptr(),n,5120,2,
                    torch.cuda.current_stream().cuda_stream)==0
        torch.cuda.synchronize();serial=None
        for lane,stream in enumerate(streams):
            with torch.cuda.stream(stream),torch.no_grad():
                bytes=workspace_bytes(max(widths),5120,n)
                workspace=torch.empty(bytes,device='cuda',dtype=torch.uint8)
                def run(start,rows):
                    out=torch.empty(rows,n,device='cuda',dtype=torch.float32)
                    assert linear(x[start:start+rows].data_ptr(),packed.data_ptr(),scales.data_ptr(),
                                  out.data_ptr(),1,rows,5120,n,workspace.data_ptr(),bytes,stream.cuda_stream)==0
                    return out
                current=torch.cat([run(i,1) for i in range(max(widths))]);stream.synchronize()
                if serial is None:serial=current
                assert torch.equal(current.view(torch.int32),serial.view(torch.int32))
                for rows in widths:
                    got=run(0,rows);stream.synchronize()
                    counts=(got.view(torch.int32)!=serial[:rows].view(torch.int32)).sum(1).tolist()
                    record={'tensor':'head.weight','output_first':first,'shape':[n,5120],'lane':lane,'rows':rows,
                            'different_elements':sum(counts),'different_by_row':counts,
                            'max_abs_error':(got-serial[:rows]).abs().max().item()}
                    results.append(record)
                    print('FAIL' if sum(counts) else 'PASS',json.dumps(record),flush=True)
        del packed,scales,serial
    assert len(results)==18
    failed=sum(r['different_elements']!=0 for r in results)
    args.output.write_text(json.dumps({'scope':'Official BF16 head converted by serving Best FP8 scales; synthetic inputs',
        'device':args.device,'name':torch.cuda.get_device_name(args.device),'widths':widths,
        'checkpoint_payload_sha256':{'head.weight':payload_hash},
        'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'results':results,'failed':failed},indent=2)+'\n')
    print(f'SUMMARY {len(results)-failed} passed; {failed} failed; {len(results)} comparisons',flush=True)
    return 1 if failed else 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-lib',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--row-invariance',action='store_true')
    mode.add_argument('--head-row-invariance',action='store_true')
    mode.add_argument('--projection-timing',action='store_true')
    mode.add_argument('--projection-algorithms',action='store_true')
    parser.add_argument('--baseline-native-lib',type=Path)
    parser.add_argument('--projection-alignment',type=int,choices=(2,16,256),default=16)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    args=parser.parse_args()
    if args.row_invariance:raise SystemExit(row_invariance(args))
    if args.head_row_invariance:raise SystemExit(head_row_invariance(args))
    if args.projection_timing:raise SystemExit(projection_timing(args))
    if args.projection_algorithms:raise SystemExit(projection_algorithms(args))
    if args.reference_dir is None:parser.error('--reference-dir required unless an invariance mode is set')
    root=Path(__file__).resolve().parents[3]
    lock=json.loads((root/'scripts/fixtures/ds41-reference-lock.json').read_text())
    source=args.reference_dir/'inference/model.py';config_path=args.reference_dir/'inference/config.json'
    for p in [source,config_path]:assert hashlib.sha256(p.read_bytes()).hexdigest()==lock['files']['inference/'+p.name]
    config=json.loads(config_path.read_text());assert config['norm_eps']==1e-20
    ns={'torch':torch,'nn':torch.nn,'ModelArgs':SimpleNamespace,
        'Linear':lambda inp,out,dtype=None:torch.nn.Linear(inp,out,bias=False,dtype=dtype or torch.bfloat16)}
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name in ['Compressor','RMSNorm']]
    assert len(nodes)==2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    index=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    payload_hashes={}
    def load(name):
        path=args.snapshot/index[name]
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length));entry=header[name]
            assert entry['dtype']=='BF16'
            a,b=entry['data_offsets'];f.seek(8+length+a);data=f.read(b-a);assert len(data)==b-a
        payload_hashes[name]=hashlib.sha256(data).hexdigest()
        return torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(entry['shape']).cuda()
    torch.cuda.set_device(args.device);torch.manual_seed(410233)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    lib=C.CDLL(str(args.native_lib));P=C.c_void_p;I=C.c_int32
    def bind(name,types):
        fn=getattr(lib,name);fn.argtypes=types;fn.restype=I;return fn
    create=bind('cuteafd_v41_compressor_create',[P,C.c_uint64,C.POINTER(P)])
    destroy=bind('cuteafd_v41_compressor_destroy',[P])
    project=bind('cuteafd_v41_compressor_project',[P,P,P,P,I,I,P])
    pool=bind('cuteafd_v41_compressor_pool',[P,P,P,P,P,P,P,I,I,P])
    norm=bind('cuteafd_v41_attention_norm',[P,P,P,P,I,I,P])
    ptr=lambda t:t.data_ptr()
    def check(status):assert status==0,status
    workspace=torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8);handle=P()
    check(create(ptr(workspace),workspace.numel(),C.byref(handle)))
    stream=torch.cuda.Stream();results=[]
    try:
        with torch.cuda.stream(stream),torch.no_grad():
            pool_results=[]
            for rows in [1,16,80,4096]:
                slots=16
                kv=torch.randn((rows,512),device='cuda');score=torch.randn_like(kv)*80
                pk=torch.randn((slots,512),device='cuda');ps=torch.randn_like(pk)*80
                weight=(torch.randn(512,device='cuda')*.2+1).bfloat16();out=torch.empty((rows,512),device='cuda',dtype=torch.bfloat16)
                desc=torch.tensor([0 if row==0 else (slots+row-1 if row%2 else (1<<64)-1) for row in range(rows)],device='cuda',dtype=torch.uint64)
                def launch_pool():check(pool(ptr(kv),ptr(score),ptr(pk),ptr(ps),ptr(desc),ptr(weight),ptr(out),rows,slots,stream.cuda_stream))
                def pool_oracle():
                    indices=desc.long();valid=(indices>=0)&(indices<slots+torch.arange(rows,device='cuda'))
                    indices=indices.clamp(0,slots+rows-1)
                    pair=torch.stack([torch.cat([pk,kv])[indices],kv],1)
                    gates=torch.stack([torch.cat([ps,score])[indices],score],1).softmax(1)
                    pooled=(pair*gates).sum(1).bfloat16().float()
                    expected=(pooled*torch.rsqrt(pooled.square().mean(-1,keepdim=True)+config['norm_eps'])*weight.float()).bfloat16()
                    expected[~valid]=0
                    torch.testing.assert_close(out,expected,rtol=.008,atol=.002)
                    return (out.float()-expected.float()).abs().max().item()
                launch_pool();stream.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):launch_pool()
                errors=[]
                for case in range(3):
                    if case==1:kv.mul_(1e-10);pk.mul_(1e-10);score.zero_();ps.zero_()
                    if case==2:kv.zero_();pk.zero_()
                    graph.replay();errors.append(pool_oracle())
                # Inputs, including committed state, are untouched by a replay.
                snapshot=[kv.clone(),score.clone(),pk.clone(),ps.clone()]
                graph.replay()
                assert all(torch.equal(x,y) for x,y in zip([kv,score,pk,ps],snapshot))
                for invalid in [(1<<64)-1,1<<40,slots+rows]:
                    desc.fill_(invalid);graph.replay();assert torch.count_nonzero(out).item()==0
                assert pool(ptr(kv),ptr(score),ptr(pk),ptr(ps),ptr(desc),ptr(weight),ptr(kv),rows,slots,stream.cuda_stream)!=0
                assert pool(ptr(kv),ptr(score),ptr(pk),ptr(ps),ptr(desc),ptr(weight),ptr(out),rows,17,stream.cuda_stream)!=0
                stream.synchronize();graph.reset()
                pool_results.append({'rows':rows,'extreme_equal_tiny_zero_errors':errors,'invalid_descriptors_zero':True,'overlap_and_slot_guards':True})
            for layer_id in [2,8,14,20]:
                ratio=config['compress_ratios'][layer_id]
                weights={name:load(f'layers.{layer_id}.attn.compressor.{name}.weight') for name in (['wkv','wgate','norm'] if ratio==2 else ['wkv','norm'])}
                # Constructors get the actual model epsilon and precision modes.
                reference=ns['Compressor'](SimpleNamespace(**{**config,'max_batch_size':16}),layer_id).cuda()
                reference.norm.weight.data=weights['norm'].clone()
                reference.wkv.weight.copy_(weights['wkv'])
                if ratio==2:reference.wgate.weight.copy_(weights['wgate'])
                for requests in [1,3,16]:
                    for chunks in [[1]*9,[1,2,5,64,127]]:
                        length=sum(chunks)
                        x=(torch.randn((requests,length,5120),device='cuda')*.125).bfloat16()
                        # Small inputs detect epsilon and pre-normalization BF16 boundaries.
                        x[:,::13]*=1e-8
                        expected=reference(x,0).clone()
                        pending_kv=torch.zeros((requests,512),device='cuda')
                        pending_scores=torch.zeros_like(pending_kv)
                        start=0;max_error=0.;projection_error=0.;different=0
                        for size in chunks:
                            inp=x[:,start:start+size].contiguous().reshape(-1,5120);rows=inp.shape[0]
                            projected=torch.empty((rows,512),device='cuda',dtype=torch.float32 if ratio==2 else torch.bfloat16)
                            scores=torch.empty((rows,512),device='cuda')
                            output=torch.empty((rows,512),device='cuda',dtype=torch.bfloat16)
                            descriptors=[];emitted=[]
                            for r in range(requests):
                                for j in range(size):
                                    pos=start+j;row=r*size+j
                                    descriptors.append((r if j==0 else requests+row-1) if pos%2 else (1<<64)-1)
                                    if ratio==1 or pos%2:emitted.append((row,r,pos//ratio))
                            pred=torch.tensor(descriptors,device='cuda',dtype=torch.uint64)
                            before=[pending_kv.clone(),pending_scores.clone()]
                            def launch():
                                check(project(handle,ptr(inp),ptr(weights['wkv']),ptr(projected),rows,ratio,stream.cuda_stream))
                                if ratio==2:
                                    check(project(handle,ptr(inp),ptr(weights['wgate']),ptr(scores),rows,ratio,stream.cuda_stream))
                                    check(pool(ptr(projected),ptr(scores),ptr(pending_kv),ptr(pending_scores),ptr(pred),ptr(weights['norm']),ptr(output),rows,requests,stream.cuda_stream))
                                else:check(norm(ptr(projected),ptr(weights['norm']),None,ptr(output),rows,512,stream.cuda_stream))
                            launch();stream.synchronize();graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=stream):launch()
                            graph.replay();first=output.clone();graph.replay()
                            assert torch.equal(output,first)
                            assert torch.equal(pending_kv,before[0]) and torch.equal(pending_scores,before[1])
                            # Compare FP32 dot to FP64 accumulation to separate GEMM reduction error.
                            dot=(inp.double() @ weights['wkv'].double().T).float()
                            if ratio==2:
                                torch.testing.assert_close(projected,dot,rtol=2e-5,atol=1e-4)
                                projection_error=max(projection_error,(projected-dot).abs().max().item())
                                gate_dot=(inp.double() @ weights['wgate'].double().T).float()
                                torch.testing.assert_close(scores,gate_dot,rtol=2e-5,atol=1e-4)
                                # Isolate pooling/norm from projection reduction differences.
                                for row,r,pos in emitted:
                                    old=pred[row].item()
                                    ka=pending_kv[old] if old<requests else projected[old-requests]
                                    sa=pending_scores[old] if old<requests else scores[old-requests]
                                    pooled=(torch.stack([ka,projected[row]])*torch.stack([sa,scores[row]]).softmax(0)).sum(0).bfloat16()
                                    oracle=reference.norm(pooled)
                                    torch.testing.assert_close(output[row],oracle,rtol=.008,atol=.002)
                            else:torch.testing.assert_close(projected,dot.bfloat16(),rtol=.008,atol=.002)
                            for row,r,pos in emitted:
                                oracle=expected[r,pos]
                                torch.testing.assert_close(output[row],oracle,rtol=.016,atol=.004)
                                max_error=max(max_error,(output[row].float()-oracle.float()).abs().max().item())
                                different+=(output[row]!=oracle).sum().item()
                            if ratio==2:
                                incomplete=[r*size+j for r in range(requests) for j in range(size) if (start+j)%2==0]
                                if incomplete:assert torch.count_nonzero(output[incomplete]).item()==0
                                # Explicit commit only after accepted execution, never during capture/replay.
                                if (start+size)%2:
                                    for r in range(requests):pending_kv[r].copy_(projected[(r+1)*size-1]);pending_scores[r].copy_(scores[(r+1)*size-1])
                                # Invalid device descriptors fail closed even on graph replay.
                                pred.fill_((1<<64)-1);graph.replay();assert torch.count_nonzero(output).item()==0
                            assert project(handle,ptr(inp),ptr(weights['wkv']),ptr(inp),rows,ratio,stream.cuda_stream)!=0
                            assert project(handle,ptr(inp),ptr(weights['wkv']),ptr(projected),0,ratio,stream.cuda_stream)!=0
                            stream.synchronize();graph.reset();start+=size
                        assert start==length
                        results.append({'layer':layer_id,'ratio':ratio,'requests':requests,'chunks':chunks,'reference_max_abs_error':max_error,'different_bf16_elements':different,'projection_fp64_max_abs_error':projection_error,'state_unchanged_until_commit':True,'graph_replay_exact':True})
                        print('PASS',results[-1],flush=True)
            stream.synchronize()
    finally:
        stream.synchronize();check(destroy(handle))
    assert len(results)==24 and len(pool_results)==4
    args.output.write_text(json.dumps({'scope':'Real checkpoint compressor weights with synthetic activations; native primitives, no Rust request state or serving','device':args.device,'name':torch.cuda.get_device_name(args.device),'reference_revision':lock['revision'],'norm_eps':config['norm_eps'],'checkpoint_payload_sha256':payload_hashes,'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pool_results':pool_results,'results':results},indent=2)+'\n')

if __name__=='__main__':main()
