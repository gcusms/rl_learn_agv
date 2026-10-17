"""Create an isolated CPU optimization handoff; never overwrites the parent."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'exports/realtest_20261016'
OUTPUT = ROOT / 'exports/realtest_edge_20261016'
REPORT = ROOT / 'reports/embedded_runtime_20261016'
PARENT_ARCHIVE = ROOT.parent / 'realtest_20261016.tar.gz'
ARCHIVE = ROOT.parent / 'realtest_edge_20261016.tar.gz'
PARENT_SHA = 'c4f4c52ce6ec852b88ce61499c889e782884753ecd44be27aaafec918620e3be'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def archived_files(root):
    return sorted(p for p in root.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'generated' not in p.relative_to(root).parts
        and p.name != 'SHA256SUMS.json')


def verify_parent():
    assert digest(PARENT_ARCHIVE) == PARENT_SHA, 'Delivered parent archive changed'
    checksums = json.loads((PARENT / 'SHA256SUMS.json').read_text())
    for name, expected in checksums.items():
        assert digest(PARENT / name) == expected, name
    return len(checksums)


def stage():
    parent_count = verify_parent()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    shutil.copytree(PARENT, OUTPUT, ignore=shutil.ignore_patterns('generated', '__pycache__'))
    for relative in ['training/neural_navigation/directional_risk.py',
            'training/neural_navigation/directional_risk_batch.py',
            'training/experiments/test_directional_risk_batch.py',
            'training/test_directional_risk.py', 'training/test_predictive_constraint.py',
            'tools/benchmark_embedded_runtime.py']:
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copytree(REPORT, OUTPUT / 'reports/embedded_runtime_20261016')
    old_readme = (OUTPUT / 'README.md').read_text().replace(
        '/absolute/path/to/realtest_20261016', '/absolute/path/to/realtest_edge_20261016')
    (OUTPUT / 'REALTEST_GUIDE.md').write_text(old_readme)
    (OUTPUT / 'README.md').write_text('''# Orin Nano / 树莓派 CPU 优化验证包

按 Orin Nano 8GB 与树莓派 5 的目标整理。本版仅优化方向风险预测的计算实现，沿用上一实车候选的模型权重和保护合同；新增训练 0 步。

当前 Xeon 主机配对回放：CPU P50 **77.74 → 54.49 ms**，P95 **172.00 → 102.22 ms**；144 帧观测、动作与执行命令一致，25 项功能检查通过。没有板端速度、整机内存或功耗结论。历史 11/12 闭环成绩来自优化前，新版仍须完整闭环与实车验证，部署验收未通过。

先读 [CPU 优化结果和板端验证步骤](reports/embedded_runtime_20261016/README.md)，再按 [ROS 配置、录制与回退说明](REALTEST_GUIDE.md)进行受控测试。ARM64 必须先用包内工具重建定位库，不能直接使用归档的 x86_64 `.so`。

在解压目录执行 `python3 verify.py` 检查文件；板端原生库和本机配置生成在 `generated/`，不覆盖归档权重。目录包含匹配的 625 维源码、模型、原 300 维实车回退包、测时工具、记录与地图，不含训练环境或 ROS 安装。

本包是本地独立交付，未自动推送远程。原 `realtest_20261016.tar.gz` 保留；`SOURCE.json` 记录父包及代码身份，`SHA256SUMS.json` 记录文件散列。定位精度、长时间等待和极窄脱困没有因本次加速而被证明改善。
''')
    source = json.loads((PARENT / 'SOURCE.json').read_text())
    source.update(created_at=datetime.now(timezone.utc).isoformat(),
        package_variant='realtest_edge_20261016',
        parent_source_sha256=digest(PARENT / 'SOURCE.json'),
        parent_archive_sha256=PARENT_SHA, parent_archived_files_verified=parent_count,
        code_sha256={str(p.relative_to(OUTPUT)): digest(p) for p in archived_files(OUTPUT)
            if p.suffix in ('.py', '.cpp', '.h', '.hpp')},
        this_variant_new_training_steps=0, this_variant_physics_episodes=0,
        historical_behavior_results_predate_optimization=True,
        optimized_runtime_report='reports/embedded_runtime_20261016/paired_host/RESULTS.json',
        target_boards_measured=False, weights_changed=False, protection_contract_changed=False)
    write(OUTPUT / 'SOURCE.json', source)
    write(OUTPUT / 'SHA256SUMS.json', {str(p.relative_to(OUTPUT)): digest(p) for p in archived_files(OUTPUT)})
    print(json.dumps(dict(staged=str(OUTPUT), parent_verified=True), indent=2))


def seal():
    verify_parent()
    if ARCHIVE.exists():
        raise FileExistsError(ARCHIVE)
    assert OUTPUT.is_dir()
    receipt = json.loads((OUTPUT / 'reports/embedded_runtime_20261016/PACKAGE_CHECKS.json').read_text())
    assert receipt['passed'], 'Run the packaged checks before sealing'
    source = json.loads((OUTPUT / 'SOURCE.json').read_text())
    for name, expected in source['code_sha256'].items():
        assert digest(OUTPUT / name) == expected, 'Staged source changed: ' + name
    models = [p for p in (PARENT / 'models').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for path in models:
        assert digest(path) == digest(OUTPUT / path.relative_to(PARENT)), path
    checksums = {str(p.relative_to(OUTPUT)): digest(p) for p in archived_files(OUTPUT)}
    write(OUTPUT / 'SHA256SUMS.json', checksums)
    with tarfile.open(ARCHIVE, 'w:gz', compresslevel=3) as tar:
        def include(info):
            parts = Path(info.name).parts
            return None if 'generated' in parts or '__pycache__' in parts else info
        tar.add(OUTPUT, arcname=OUTPUT.name, filter=include)
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        archived_sums = json.load(tar.extractfile(OUTPUT.name + '/SHA256SUMS.json'))
        assert archived_sums == checksums
        for name, expected in checksums.items():
            assert hashlib.sha256(tar.extractfile(OUTPUT.name + '/' + name).read()).hexdigest() == expected, name
        archived_names = {m.name for m in tar.getmembers() if m.isfile()}
        assert archived_names == {OUTPUT.name + '/' + n for n in [*checksums, 'SHA256SUMS.json']}
    archive_sha = digest(ARCHIVE)
    ARCHIVE.with_name(ARCHIVE.name + '.sha256').write_text(archive_sha + '  ' + ARCHIVE.name + '\n')
    result = dict(completed=True, archive=str(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
        archive_sha256=archive_sha, archived_files_verified=len(checksums) + 1,
        unchanged_model_files_verified=len(models), parent_archive_unchanged=True,
        source_hashes_verified=True, package_checks_passed=True,
        target_boards_measured=False, vehicle_started=False, new_training_steps=0)
    write(REPORT / 'ARCHIVE_CHECKS.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true', help='Seal a staged and checked package')
    args = parser.parse_args()
    seal() if args.seal else stage()
