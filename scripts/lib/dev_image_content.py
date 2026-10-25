#!/usr/bin/env python3
"""A dev image's portable content identity, from its `docker image inspect` record.

The image ID is not portable: docker save | load into a containerd image store reports
the OCI manifest digest where the classic store reports the config digest, for
byte-identical layers and config. This digest covers what a container runs: platform,
run config (labels, so the toolchain hash; env, entrypoint, user, ...) and the RootFS
diff IDs. Store-specific fields (size, digests, graph driver, empty defaults) are left out.

  docker image inspect IMAGE | scripts/lib/dev_image_content.py   # prints the digest
"""
import hashlib
import json
import sys


def _unset(value):
    # Fields one image store emits as an empty default and another omits: null, false,
    # '' and empty lists/maps. Numbers are kept: StopTimeout 0 differs from no StopTimeout.
    return value is None or value is False or (isinstance(value, (str, list, dict)) and not value)


def image_content(info):
    config = {k: v for k, v in sorted((info.get('Config') or {}).items()) if not _unset(v)}
    rootfs = info.get('RootFS') or {}
    body = {'architecture': info.get('Architecture'), 'os': info.get('Os'), 'variant': info.get('Variant') or '',
            'config': config, 'rootfs_type': rootfs.get('Type'), 'layers': rootfs.get('Layers') or []}
    if not body['layers'] or not body['architecture'] or not body['os']:
        return None
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    try:
        records = json.load(sys.stdin)
    except ValueError as error:
        sys.exit(f'dev_image_content: not docker image inspect JSON: {error}')
    if not isinstance(records, list) or len(records) != 1:
        sys.exit('dev_image_content: expected exactly one image record')
    digest = image_content(records[0])
    if not digest:
        sys.exit('dev_image_content: the image has no RootFS layers or platform')
    print(digest)


if __name__ == '__main__':
    main()
