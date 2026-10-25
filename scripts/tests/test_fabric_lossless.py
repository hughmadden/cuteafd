"""CPU-only fabric configuration guards; no NIC or switch writes."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_traffic_class_parser_and_roce_sl(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'traffic.cc'
    source.write_text(r'''
#include "rdma_traffic_class.h"
#include <cassert>
int main() {
  uint8_t tc = 99;
  assert(cuteafd::rdma_traffic_class(nullptr, &tc) && tc == 0);
  assert(cuteafd::rdma_traffic_class("", &tc) && tc == 0);
  assert(cuteafd::rdma_traffic_class("104", &tc) && tc == 104);
  assert((tc >> 2) == 26 && (tc >> 5) == 3);
  assert(cuteafd::rdma_traffic_class("255", &tc) && tc == 255);
  for (const char* bad : {"-1", "+1", "256", "26x", " 104", "999999999999999"})
    assert(!cuteafd::rdma_traffic_class(bad, &tc));
}
'''.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>'))
    executable = tmp_path / 'traffic'
    subprocess.run([compiler, '-std=c++17', '-I', str(ROOT / 'native/shared/src'),
                    str(source), '-o', str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
    native = (ROOT / 'native/shared/src/cuteafd_native.cc').read_text()
    assert 'std::getenv("CUTEAFD_RDMA_TRAFFIC_CLASS")' in native
    assert '&attr.ah_attr.grh.traffic_class' in native
    assert 'attr.ah_attr.sl = attr.ah_attr.grh.traffic_class >> 5;' in native


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
