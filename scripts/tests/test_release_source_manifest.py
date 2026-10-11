import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'build' / 'verify-release-source-manifest.py'
spec = importlib.util.spec_from_file_location('verify_release_source_manifest', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def tree(tmp_path):
    source = tmp_path / 'src'
    (source / 'dep' / '.agents' / 'skills').mkdir(parents=True)
    (source / 'dep' / '.agents' / 'skills' / 'a.md').write_text('a')
    (source / 'dep' / '.claude').mkdir()
    return source


def test_in_tree_directory_symlink_is_skipped_and_target_inventoried(tmp_path):
    source = tree(tmp_path)
    (source / 'dep' / '.claude' / 'skills').symlink_to('../.agents/skills', target_is_directory=True)
    assert module.source_inventory(source) == {'./dep/.agents/skills/a.md'}


@pytest.mark.parametrize('target, reason', [('../../../outside', 'escapes'), ('../missing', 'escapes')])
def test_escaping_or_broken_directory_symlink_is_refused(tmp_path, target, reason):
    source = tree(tmp_path)
    (tmp_path / 'outside').mkdir()
    (source / 'dep' / '.claude' / 'skills').symlink_to(target, target_is_directory=True)
    with pytest.raises(module.SourceManifestError, match=reason):
        module.source_inventory(source)
