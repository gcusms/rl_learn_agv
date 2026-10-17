"""Archive this bounded real-car candidate; never trains or publishes a model."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REPORTS = Path('reports/stop_feedback_20261015')
MODEL = Path('models/stop_feedback_20261015/release_context_bc2000_s231016')
PARENT = Path('models/stop_feedback_20261015/long_risk_bc1000_s231016')
BASELINE = Path('models/quality_recovery_20261014')
WORKSPACE = ROOT.parent
RECORD = WORKSPACE / 'record_analysis_20261014/raw/20261014_193644_map_01test_953a9959/snapshots'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    result = json.loads((ROOT / REPORTS / 'release_validation_20261016/RESULTS.json').read_text())
    assert result['completed'] and len(result['rows']) == 12
    assert sum(row['safe_success'] for row in result['rows']) == 11
    state = subprocess.check_output(['systemctl', '--user', 'show',
        'rl-release-validation-20261016.service', '-p', 'MainPID', '--value'], text=True).strip()
    assert state == '0', 'Do not archive changing validation inputs/results'
    assert digest(ROOT / MODEL / 'checkpoint.pt') == 'a77684103768dbdaa5213246118e05b74fc108cc93f9ed25978dd48da2b0cfe3'
    assert digest(ROOT / BASELINE / 'policy_2000.zip') == '7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11'
    output = ROOT / 'exports/realtest_20261016'
    output.mkdir(parents=True, exist_ok=False)

    def copy(source, target=None):
        source = Path(source)
        source = source if source.is_absolute() else ROOT / source
        target = output / (target if target is not None else source.relative_to(ROOT))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # Full inference code, lightweight experiment implementations, no caches.
    for directory in ['training/neural_navigation', 'training/navrl', 'training/experiments']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.cpp', '.h', '.hpp'):
                copy(path)
    for directory in [MODEL, PARENT]:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                copy(path)
    for path in (ROOT / BASELINE / 'portable').iterdir():
        if path.is_file():
            copy(path)
    copy(BASELINE / 'policy_2000.zip')
    for path in (ROOT / 'deployment/ros2_recording_adapter').glob('*'):
        if path.is_file() and path.suffix in ('.py', '.md'):
            copy(path)
    patch_names = ['model_node.py', 'inference_worker.py', 'execution_feedback.py', 'model_bundle.py']
    for path in (RECORD / 'package/rl_navigation').rglob('*.py'):
        copy(path, Path('reference_ros/rl_navigation') / path.relative_to(RECORD / 'package/rl_navigation'))
    for name in patch_names:
        source = Path('deployment/ros2_recording_adapter') / name
        copy(source, 'ros2_patch/rl_navigation/' + name)
        copy(source, 'reference_ros/rl_navigation/' + name)
    for name in ['map.yaml', 'map.pgm', 'original.yaml', 'source.json']:
        copy(RECORD / 'config/referenced/map_server/map' / name, 'maps/recorded_map_01/' + name)
    copy(RECORD / 'package/config/chassis.yaml', 'reference_ros/recorded_chassis.yaml')
    copy(RECORD / 'runs/5113962c8a9f49c6a665a14ea96902a6/metadata.json', 'reference_ros/recorded_session_metadata.json')
    for name in ['configure.py', 'verify.py', 'smoke_adapter.py']:
        copy('deployment/realtest_tools/' + name, name)
    copy('tools/prepare_native_bundle.py')
    for relative in ['agv/simulations/agv_description/urdf/agv_chassis.urdf']:
        copy(relative)
    folders = ['release_validation_20261016', 'release_sources_20261016',
        'long_risk_validation_20261016', 'long_risk_wait_audit_20261016',
        'release_broad_20261016', 'release_decision_audit_20261016']
    for folder in folders:
        for path in sorted((ROOT / REPORTS / folder).rglob('*')):
            if path.is_file() and 'source_snapshot' not in path.parts and '__pycache__' not in path.parts:
                copy(path)
    case_paths = ['proactive_augmented_protocol_20261016/fresh_cases.json',
        'early_yield_remaining_protocol_20261016/heldout_cases.json',
        'long_risk_protocol_20261016/new_sensor_cases.json']
    for name in case_paths:
        copy(REPORTS / name)
        for case in json.loads((ROOT / REPORTS / name).read_text()):
            # Preserve the exact raster/map for sensor replay. This does not
            # turn this inference archive into a complete MuJoCo installation.
            scene = ROOT / 'training' / case['scene']
            for path in scene.iterdir():
                if path.is_file() and path.suffix in ('.yaml', '.pgm', '.png'):
                    copy(path)
    table = ['| 仿真组 | 到达 | 碰撞 | 超时 |', '|---|---:|---:|---:|']
    for key, label in [('repaired_wait_condition', '旧等待失败条件'),
        ('retained_development', '保留的成功条件'), ('frozen_new_noise', '新冻结噪声条件')]:
        rows = [row for row in result['rows'] if row['group'] == key]
        table.append(f"| {label} | {sum(r['safe_success'] for r in rows)}/{len(rows)} | "
            f"{sum(r['collisions'] for r in rows)} | {sum(r['timeout'] for r in rows)} |")
    readme = (ROOT / 'deployment/realtest_tools/README.md').read_text().replace('@@RESULT_TABLE@@', '\n'.join(table))
    (output / 'README.md').write_text(readme)
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'scipy', 'PyYAML', 'Pillow']}
    write(output / 'ENVIRONMENT.json', dict(python=sys.version, platform=platform.platform(),
        machine=platform.machine(), inference_packages=versions,
        not_a_jetson_install_lockfile=True, ros_node_executed=False))
    code = {str(path.relative_to(output)): digest(path) for path in output.rglob('*')
        if path.is_file() and path.suffix in ('.py', '.cpp', '.h', '.hpp')}
    write(output / 'SOURCE.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_worktree=str(ROOT), base_git_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        includes_uncommitted_changes=True, new_research_pushed_to_github=False,
        fixed_github_branch='handoff3', fixed_github_commit='6d443f31bb3b20d2a5d31539a6490be3c2bddf05',
        code_sha256=code, checkpoint_sha256=digest(output / MODEL / 'checkpoint.pt'),
        original_policy_sha256=digest(output / BASELINE / 'policy_2000.zip'),
        actual_supervised_updates=50, new_ppo_steps=0, safe_success=11, physical_total=12,
        full_regression_completed=False, deployment_approved=False, vehicle_started=False,
        includes_all_historical_training_dependencies=False))
    checksums = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob('*')) if path.is_file()}
    write(output / 'SHA256SUMS.json', checksums)
    if '--stage-only' in sys.argv:
        print(json.dumps(dict(directory=str(output), files=len(checksums), staged=True), indent=2))
        return
    archive = WORKSPACE / 'realtest_20261016.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        tar.add(output, arcname=output.name)
    (archive.parent / (archive.name + '.sha256')).write_text(digest(archive) + '  ' + archive.name + '\n')
    print(json.dumps(dict(directory=str(output), archive=str(archive), files=len(checksums),
        archive_bytes=archive.stat().st_size, archive_sha256=digest(archive)), indent=2))


if __name__ == '__main__':
    main()
