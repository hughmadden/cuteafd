"""Historical evidence cannot pass through changed numerics or missing proofs."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm5_flash.core_k6.pilot import (canonical, decision_policy, validate_policy_change,
    validate_report_code)
from glm5_flash.core_k6.finalize import native_checks_from_events


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_policy_ast_audit_allows_reporting_only_and_rejects_capture_or_global_changes():
    original='''RECIPE = 6
def replay(x):
    return x + 1
def validate_qualification(report):
    return report["fp8"]
'''
    current=original.replace('return report["fp8"]','return report["absolute"]')+'\ndef decision_policy():\n    return .001\n'
    assert "validate_qualification" in validate_policy_change(original,current,digest(original))
    with pytest.raises(ValueError,match="capture/scoring/replay"):
        validate_policy_change(original,current.replace("x + 1","x + 2"),digest(original))
    with pytest.raises(ValueError,match="capture/scoring/replay"):
        validate_policy_change(original,current.replace("RECIPE = 6","RECIPE = 5"),digest(original))
    with pytest.raises(ValueError,match="Git source"):
        validate_policy_change(original,current,"0"*64)


def fixture(tmp_path,monkeypatch):
    repo=tmp_path/"repo";repo.mkdir()
    root=tmp_path/"evidence";root.mkdir()
    relative="quantization/glm5_flash/core_k6/pilot.py"
    path=repo/relative;path.parent.mkdir(parents=True)
    original='def replay(x):\n    return x + 1\ndef validate_qualification(r):\n    return r\n'
    current=original.replace('return r\n','return r.copy()\n')
    path.write_text(current)
    finalizer=path.with_name("finalize.py");finalizer.write_text('reporting_only = True\n')
    numerical=repo/"numerical.py";numerical.write_text('native = 1\n')
    identity=dict(gptqmodel_commit="f"*40,code={relative:digest(original),"numerical.py":digest(numerical.read_text())})
    (root/"plan.json").write_text(json.dumps(identity))
    checks=dict(native=True)
    policy_hash=digest(canonical(decision_policy()))
    audit=dict(status="passed",identity=identity,native_checks=checks,decision_policy_sha256=policy_hash)
    (root/"finalization-audit.json").write_text(json.dumps(audit))
    migration=dict(schema="cuteafd-flash-pilot-report-only-migration-v1",original_pilot_git_revision="a"*40,
        original_pilot_sha256=digest(original),current_pilot_sha256=digest(current),
        allowed_reporting_nodes=validate_policy_change(original,current,digest(original)),
        decision_policy_sha256=policy_hash,original_plan_sha256=digest((root/"plan.json").read_text()),
        audit_file="finalization-audit.json",audit_sha256=digest((root/"finalization-audit.json").read_text()),
        finalizer_code_sha256=digest(finalizer.read_text()))
    report=dict(identity=identity,native_checks=checks,evidence_root=str(root),policy_migration=migration,
        decision_policy=decision_policy())
    def git(args,**kwargs):
        return original if "show" in args else identity["gptqmodel_commit"]
    monkeypatch.setattr("glm5_flash.core_k6.pilot.subprocess.check_output",git)
    return repo,root,report


def test_report_code_binds_numerics_policy_original_plan_audit_and_finalizer(tmp_path,monkeypatch):
    repo,root,report=fixture(tmp_path,monkeypatch)
    assert validate_report_code(report,repo) is report
    (repo/"numerical.py").write_text('native = 2\n')
    with pytest.raises(ValueError,match="numerical producer changed"):
        validate_report_code(report,repo)


def test_report_code_rejects_forged_migration_for_changed_replay(tmp_path,monkeypatch):
    repo,root,report=fixture(tmp_path,monkeypatch)
    pilot=repo/"quantization/glm5_flash/core_k6/pilot.py"
    pilot.write_text(pilot.read_text().replace("x + 1","x + 2"))
    report["policy_migration"]["current_pilot_sha256"]=digest(pilot.read_text())
    with pytest.raises(ValueError,match="capture/scoring/replay"):
        validate_report_code(report,repo)


def test_report_code_rejects_tampered_audit_or_finalizer(tmp_path,monkeypatch):
    repo,root,report=fixture(tmp_path,monkeypatch)
    finalizer=repo/"quantization/glm5_flash/core_k6/finalize.py"
    finalizer.write_text('reporting_only = False\n')
    with pytest.raises(ValueError,match="finalizer implementation"):
        validate_report_code(report,repo)
    finalizer.write_text('reporting_only = True\n')
    (root/"finalization-audit.json").write_text('{}')
    with pytest.raises(ValueError,match="audit hash"):
        validate_report_code(report,repo)


def event_fixture():
    cal=[dict(id="first",input_ids=[0]*723),dict(id="long",input_ids=[0]*2205)]
    held=[dict(id="held",input_ids=[0]*780)]
    def observed(kind,layer,row):
        n=len(row["input_ids"])
        indexer=None if layer not in (3,7) else dict(byte_exact=True,tokens=n,raw_topk=2048,pool_size=4,
            pool_budget=512,valid_last_row=min(n,2049),sparse_selection=n>2049)
        return dict(event=kind,layer=layer,row_id=row["id"],tokens=n,indexer=indexer)
    def events(records):
        return [e for layer in range(8) for e in [dict(event="layer_committed",layer=layer)]+
                [observed("native_boundary_exact",layer,row) for row in records]]
    root=[observed("serialized_boundary_exact",layer,row) for layer in (1,4,7) for row in cal]
    root.append(dict(event="mtp_boundary_exact",tokens=723,shifted_pairs=722))
    mtp=dict(byte_exact=True,serialized_k4_byte_exact=True,prefix_layers=8,final_target_frontier=False,
        source_row="first",tokens=723,shifted_pairs=722,projection_evidence=dict(bits=4,identity_rows=0,hessian_rows=722),
        projection_metrics=dict(hessian_sample_count=722,quantizer_path="hessian_ldlq",hessian_metric_status="ok"),
        contract=dict(positions="shifted-token-position[t+1]"))
    return dict(identity=dict(long_context_witness="long"),calibration=cal,heldout=held,root_events=root,
                cal_events=events(cal),held_events=events(held),mtp=mtp)


def test_native_checks_require_actual_complete_measured_events():
    args=event_fixture()
    checks,observed=native_checks_from_events(**args)
    assert all(checks.values()) and len(observed)==6
    args["root_events"].pop()
    with pytest.raises(ValueError,match="measured mtp_boundary_exact"):
        native_checks_from_events(**args)


def test_native_checks_reject_missing_native_layer_or_false_sparse_observation():
    args=event_fixture();args["cal_events"].pop(0)
    with pytest.raises(ValueError,match="layer_committed"):
        native_checks_from_events(**args)
    args=event_fixture()
    row=next(r for r in args["cal_events"] if r.get("layer")==7 and r.get("row_id")=="long")
    row["indexer"]["byte_exact"]=False
    with pytest.raises(ValueError,match="pooled-indexer proof"):
        native_checks_from_events(**args)
