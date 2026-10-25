"""CPU-only fabric configuration guards; no NIC or switch writes."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_marks_every_rc_qp_dscp26_priority3(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'traffic.cc'
    source.write_text(r'''
#include "rdma_traffic_class.h"
static_assert(cuteafd::kRdmaTrafficClass == 104, "DSCP 26, Not-ECT");
static_assert((cuteafd::kRdmaTrafficClass >> 2) == 26 && (cuteafd::kRdmaTrafficClass >> 5) == 3, "SL 3");
int main() {}
''')
    subprocess.run([compiler, '-std=c++17', '-I', str(ROOT / 'native/shared/src'), str(source),
                    '-o', str(tmp_path / 'traffic')], check=True, capture_output=True)
    native = (ROOT / 'native/shared/src/cuteafd_native.cc').read_text()
    assert 'getenv("CUTEAFD_RDMA_TRAFFIC_CLASS")' not in native
    assert 'attr.ah_attr.grh.traffic_class = cuteafd::kRdmaTrafficClass;' in native
    assert 'attr.ah_attr.sl = cuteafd::kRdmaTrafficClass >> 5;' in native
    launcher = (ROOT / 'scripts/launch/run-family.sh').read_text()
    assert 'RDMA_TRAFFIC_CLASS' not in launcher

def test_switch_files_keep_updated_baseline_and_separate_dscp_only():
    setup = (ROOT / 'scripts/fabric/mikrotik-crs8xx-roce-lossless.rsc').read_text()
    rollback = (ROOT / 'scripts/fabric/mikrotik-crs8xx-roce-rollback.rsc').read_text()
    for text in [setup, rollback]:
        assert 'ecn=yes' not in text and 'wred=yes' not in text
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line and not line.startswith('#'):
                assert index > 0 and lines[index - 1].startswith('#'), line
    assert setup.count('trust-l2=keep trust-l3=keep') == 7
    assert setup.count('egress-rate-queue3=100.0Gbps') == 6
    assert setup.count('egress-rate-queue3=400.0Gbps') == 1
    assert setup.count('automap=no') == 2
    assert 'map ip add dscp=26 profile=cuteafd-roce' in setup
    assert 'map ip add dscp=48 profile=cuteafd-cnp' in setup
    assert 'traffic-class=3 rx=yes tx=yes' in setup
    assert 'set 1 ecn=no schedule=low-priority-group weight=2 wred=no' in rollback
    assert rollback.count('!egress-rate-queue3') == 33
    assert rollback.count('pfc=disabled') == 32
    assert 'profile="cuteafd-roce"' in rollback
