"""Make separately hashed configuration variants; do not train or replace actors."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TARGET = ROOT / 'models/stop_feedback_20261015/memory_evidence_s231017'
CHANGE = dict(free_evidence_margin_m=.03, free_evidence_frames=3, free_evidence_max_gap_s=.4)
SOURCES = {
    'bc4500': ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable',
    'ppo1665': ROOT / 'models/stop_feedback_20261015/native625_batch_credit_s231017/portable',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not TARGET.exists(), 'Separate fresh artifact folder required'
    work = TARGET.with_name(TARGET.name + '.building')
    assert not work.exists(), 'Inspect an interrupted build before trying again'
    evidence = read(HERE / 'MEMORY_FREE_EVIDENCE_TESTS.json')
    assert evidence['passed'] and evidence['tests_run'] == 9
    for source in SOURCES.values():
        manifest = read(source / 'manifest.json')
        assert not manifest['deployment_approved']
        assert all(digest(source / name) == expected for name, expected in manifest['files'].items())
        assert read(source / 'hardware.json')['near_field_memory']['static_footprint_margin_m'] == .04
    work.mkdir(parents=True)
    rows = []
    for name, source in SOURCES.items():
        output = work / name / 'portable'
        shutil.copytree(source, output)
        hardware = read(output / 'hardware.json')
        original = read(source / 'hardware.json')
        hardware['near_field_memory'].update(CHANGE)
        compare = json.loads(json.dumps(hardware))
        for field in CHANGE:
            compare['near_field_memory'].pop(field, None)
        old_compare = json.loads(json.dumps(original))
        for field in CHANGE:
            old_compare['near_field_memory'].pop(field, None)
        assert compare == old_compare
        assert hardware['near_field_memory']['static_footprint_margin_m'] == .04
        save(output / 'hardware.json', hardware)
        preservation = read(output / 'navigation_preservation.json')
        preservation.pop('actions_masks_body_margin_blind_memory_and_deadlines_unchanged', None)
        preservation.update(memory_evidence_variant=dict(
            schema='finite_multiframe_free_evidence_v1', parameters=CHANGE,
            applied_from_initialization=True, no_episode_time_or_map_coordinate_trigger=True,
            finite_three_neighbor_ray_evidence_only=True, unknown_and_blind_points_retained=True,
            same_body_4cm_safety_margin_action_set_and_dynamics=True,
            action_mask_results_may_differ=True, frozen_actor_weights=True,
            new_ppo_steps_this_variant=0, original_actor_training_steps=preservation.get('new_ppo_steps', 0),
            calibrated_against_real_lidar=False, closed_loop_validation_required=True))
        save(output / 'navigation_preservation.json', preservation)
        manifest = read(output / 'manifest.json')
        assert 'MODEL_CARD.json' not in manifest['files']
        manifest.update(navigation_preservation_contract=preservation,
            source_configuration_manifest_sha256=digest(source / 'manifest.json'),
            exact_action_parity=False, frozen_actor_same_input_action_parity=True,
            sensor_memory_evidence_changed=True, new_ppo_steps_this_configuration=0,
            deployment_approved=False)
        manifest['files'] = {filename: digest(output / filename) for filename in manifest['files']}
        save(output / 'manifest.json', manifest)
        card = read(output / 'MODEL_CARD.json')
        card.update(manifest_sha256=digest(output / 'manifest.json'), deployment_approved=False,
            configuration_only_variant=True, new_ppo_steps_this_variant=0,
            frozen_actor_source=str(source.relative_to(ROOT)), memory_evidence_change=CHANGE,
            no_clock_gating_or_runtime_teacher=True, not_yet_closed_loop_accepted=True)
        card.setdefault('limitations', []).append(
            'Memory evidence thresholds are research settings, not real-LiDAR calibration; changed masks require full closed-loop validation.')
        save(output / 'MODEL_CARD.json', card)
        unchanged = {filename: digest(output / filename) == digest(source / filename)
            for filename in manifest['files'] if filename not in ['hardware.json', 'navigation_preservation.json']}
        assert all(unchanged.values())
        assert all(digest(output / filename) == expected for filename, expected in manifest['files'].items())
        rows.append(dict(name=name, source=str(source.relative_to(ROOT)),
            output=str((TARGET / name / 'portable').relative_to(ROOT)),
            source_manifest_sha256=digest(source / 'manifest.json'),
            manifest_sha256=digest(output / 'manifest.json'), actor_sha256=digest(output / 'actor.pt'),
            unchanged_component_hashes_verified=unchanged, original_memory=original['near_field_memory'],
            memory_change=CHANGE, new_ppo_steps=0, deployment_approved=False))
    save(work / 'VARIANTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        variants=rows, builder_sha256=digest(Path(__file__)),
        tests_sha256=digest(HERE / 'MEMORY_FREE_EVIDENCE_TESTS.json'),
        source_models_unchanged=True, no_new_physics=True, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False))
    (work / 'README.md').write_text(
        '# 连续有限回波确认的记忆配置候选\n\n'
        '这里只生成独立配置候选，新增训练0步；actor、定位器、路由器和方向停靠权重逐文件相同。'
        'bc4500与ppo1665分别对应两个原模型，不能相加其训练步数。\n\n'
        '改变的是记忆点的空闲证据阈值：连续3个不同时间戳、间隔不超过0.4秒的有限回波，'
        '且三个相邻雷达束均穿过旧点至少3厘米，才可清除。3厘米是清除证据阈值，'
        '车体碰撞余量仍为4厘米。未知、遮挡、满量程和后方盲区不提供清除证据；不按时间删除点。\n\n'
        '配置从初始化起生效，没有90/97.6/106秒等运行触发条件。'
        '此前接管时切换配置的单项收益，尚不能证明该候选完整闭环有效。'
        '所有manifest均为deployment_approved=false；未覆盖原模型或实车交付包。'
        '完整来源与哈希见VARIANTS.json。\n', encoding='utf-8')
    work.rename(TARGET)
    print(json.dumps(dict(completed=True, target=str(TARGET.relative_to(ROOT)),
        variants=[dict(name=r['name'], actor_sha256=r['actor_sha256']) for r in rows],
        new_ppo_steps=0, deployment_approved=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
