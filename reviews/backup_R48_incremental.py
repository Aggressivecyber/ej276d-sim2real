"""Normal-copy R48 delivery with explicit prior backup dependencies; no migration test."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305').resolve()
STAGE = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
RESERVE = 256 * 2**20
START = time.monotonic()

def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')

def git(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='replace')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()

assert not (ROOT / 'R48_BACKUP_RECEIPT.json').exists()
assert read(f'work/{STAGE}/FINAL_RECEIPT.json')['status'] == 'sealed'
pub = read('R48_PUBLICATION_RECEIPT.json')
assert pub['status'] == 'verified_published' and pub['match']
head = git('rev-parse', 'HEAD')
assert head == pub['local_HEAD'] == pub['GitHub_main']
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
prior = read('R47_BACKUP_RECEIPT.json')
terminal = read('R47_DELIVERY_COMPLETION_RECEIPT.json')
assert prior['status'] == 'copy_completed' and prior['copy_operations_completed_without_exception']
assert terminal['copy_operations_completed_without_exception'] and terminal['remote_SHA_match']
base = Path(prior['backup_root']).resolve()
assert base.parent == PARENT and base.is_dir()
previous = json.loads((base / 'BACKUP_FILE_LIST.json').read_text('utf-8-sig'))
assert previous['status'] == 'copy_completed'
entries = {row['path']: row for row in previous['files']}
storage_roots, previous_names = {}, {}
for old_name, location in previous['storage_roots'].items():
    path = Path(location).resolve()
    assert path.parent == PARENT and path.is_dir()
    new_name = 'R47_incremental_copy' if old_name == 'this_incremental_copy' else old_name
    storage_roots[new_name] = str(path)
    previous_names[old_name] = new_name
index = read('R48_DELIVERY_INDEX.json')
dest = Path(index['backup']['directory']).resolve()
assert dest.parent == PARENT and dest.name.startswith('r48_guided_noise_final_') and not dest.exists()
storage_roots['this_incremental_copy'] = str(dest)
program = read('PROGRAM.json')
active_stage = None
if program['current_stage'] != STAGE:
    assert (ROOT / program['current_stage_registration']).is_file()
    active_stage = f"work/{program['current_stage']}/"
skip = {'.git', '.cache', '.runtime-cache', 'runtime_cache', '__pycache__', '.venv',
        'mpl-cache', '.pytest_cache', 'node_modules'}
logical, copied, referenced, excluded = [], [], [], []
stage_bytes = 0
for current, dirs, names in os.walk(ROOT, followlinks=False):
    here = Path(current)
    keep = []
    for name in dirs:
        child = here / name
        relative = child.relative_to(ROOT).as_posix() + '/'
        if name in skip or relative == active_stage:
            excluded.append(relative)
            continue
        assert child.resolve().is_relative_to(ROOT)
        keep.append(name)
    dirs[:] = keep
    for name in names:
        source = here / name
        relative = source.relative_to(ROOT).as_posix()
        if relative == 'R48_BACKUP_RECEIPT.json':
            continue
        assert source.resolve().is_relative_to(ROOT)
        size = source.stat().st_size
        stored = 'project/' + relative
        old = entries.get(stored)
        if relative.startswith(f'work/{STAGE}/'):
            stage_bytes += size
        reuse = (relative.startswith('work/') and not relative.startswith(f'work/{STAGE}/')
                 and old is not None and old['source_bytes'] == size)
        if reuse:
            storage = previous_names[old['storage']]
            assert (Path(storage_roots[storage]) / stored).is_file()
            row = {'path': stored, 'source_bytes': size, 'storage': storage}
            referenced.append(row)
        else:
            row = {'path': stored, 'source_bytes': size, 'storage': 'this_incremental_copy'}
            copied.append((relative, size))
        logical.append(row)
assert stage_bytes <= 160 * 2**20
copy_bytes = sum(size for _, size in copied)
free_before = shutil.disk_usage(PARENT).free
assert free_before > copy_bytes + 24 * 2**20 + RESERVE, {
    'D_free': free_before, 'copy_bytes': copy_bytes, 'bundle_and_metadata_allowance': 24 * 2**20, 'reserve': RESERVE}
dest.mkdir()
print(json.dumps({'status': 'incremental_copying', 'destination': str(dest),
                  'copy_files': len(copied), 'copy_bytes': copy_bytes,
                  'referenced_files': len(referenced), 'new_stage_bytes': stage_bytes}), flush=True)
for relative, _ in copied:
    target = dest / 'project' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)
base_commit = terminal['terminal_metadata_commit']
git('merge-base', '--is-ancestor', base_commit, head)
bundle = dest / 'r48-incremental.bundle'
git('bundle', 'create', str(bundle), 'main', '^' + base_commit)
bundle_chain = [previous['bundle']['base_full_bundle'], previous['bundle']['base_metadata_bundle'],
                str(base / previous['bundle']['path']),
                str(Path(terminal['metadata_overlay_root']) / 'delivery-metadata.bundle')]
assert all(Path(p).is_file() for p in bundle_chain)
bundle_info = {'path': bundle.name, 'bytes': bundle.stat().st_size,
               'required_base_commit': base_commit, 'required_existing_bundle_chain_in_order': bundle_chain}
manifest = {'status': 'copy_completed', 'backup_mode': 'incremental_with_explicit_R44_R45_R46_R47_dependencies',
    'git_commit': head, 'storage_roots': storage_roots, 'files': sorted(logical, key=lambda row: row['path']),
    'bundle': bundle_info, 'excluded_directories': sorted(excluded),
    'content_identity_comparison_performed': False,
    'base_reference_basis': 'Completed normal-copy receipts, immutable prior-stage policy, path presence and saved size metadata; no cross-end hash test.',
    'restore_rule': 'Copy each logical project path from its named storage root into a NEW E/D destination. This directory alone is incomplete.'}
save(dest / 'BACKUP_FILE_LIST.json', manifest)
(dest / 'RESTORE_README.txt').write_text(
    'R48 incremental backup. Keep all storage roots in BACKUP_FILE_LIST.json.\n'
    'Copy each logical project file from its named storage root into a NEW E/D directory.\n'
    'For Git, restore the required_existing_bundle_chain_in_order, then r48-incremental.bundle.\n'
    'Apply final R48 delivery_completion overlay last if present, following its own receipt.\n'
    'Prior backups were neither overwritten nor deleted. Normal-copy completion, not a restore or migration identity test.\n', 'utf8')
receipt = {'status': 'copy_completed', 'backup_mode': manifest['backup_mode'],
    'completed_UTC': datetime.now(timezone.utc).isoformat(), 'source_root': str(ROOT), 'backup_root': str(dest),
    'base_backup_roots_required': [v for k, v in storage_roots.items() if k != 'this_incremental_copy'],
    'required_existing_Git_bundle_chain_in_order': bundle_chain,
    'standalone_full_snapshot': False, 'git_commit': head,
    'project_file_count': len(logical), 'project_source_bytes': sum(row['source_bytes'] for row in logical),
    'copied_file_count': len(copied), 'copied_source_bytes': copy_bytes,
    'base_referenced_file_count': len(referenced), 'base_referenced_source_bytes': sum(row['source_bytes'] for row in referenced),
    'new_stage_bytes': stage_bytes, 'copy_operations_completed_without_exception': True,
    'content_identity_tests_performed': False, 'cross_end_file_hash_comparison': False,
    'restore_execution_performed': False, 'existing_backups_deleted_or_modified': False,
    'bundle_create_native_exit': 0, 'bundle_bytes': bundle.stat().st_size, 'bundle': bundle_info,
    'file_list': str(dest / 'BACKUP_FILE_LIST.json'), 'restore_guide': str(dest / 'RESTORE_README.txt'),
    'active_stage_work_excluded': active_stage, 'external_original_data_implicitly_archived': False,
    'D_free_before_bytes': free_before, 'D_free_after_bytes': shutil.disk_usage(PARENT).free,
    'minimum_D_free_reserve_bytes': RESERVE, 'elapsed_s': time.monotonic() - START}
assert receipt['D_free_after_bytes'] >= RESERVE
for target in (ROOT / 'R48_BACKUP_RECEIPT.json', dest / 'R48_BACKUP_RECEIPT.json', dest / 'project/R48_BACKUP_RECEIPT.json'):
    assert not target.exists()
    save(target, receipt)
print(json.dumps(receipt, ensure_ascii=False), flush=True)
