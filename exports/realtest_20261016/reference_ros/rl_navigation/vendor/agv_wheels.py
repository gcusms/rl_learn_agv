#!/usr/bin/env python3
"""Standalone four-wheel command/encoder client; no ROS dependency."""
import argparse
from dataclasses import asdict
import json
import math
import signal
import sys
import time

from .ros_robot_controller_sdk import Board
from .wheel_feedback import Window, EncoderRest
from .mecanum_kinematics import MecanumKinematics

WHEEL_NAMES = ('left_front', 'left_rear', 'right_front', 'right_rear')
ZERO = [(i, 0.0) for i in range(1, 5)]


class EncoderReader:
    def __init__(self, board, geometry=None):
        self.board = board
        self.geometry = geometry
        self.window = Window(minimum_ms=20)
        self.rest = EncoderRest()
        self.resets = board.wheel_feedback_tracker.resets
        self.latest = None
        self.measurement = None
        self.measurement_at = None

    def poll(self):
        if getattr(self.board, 'recv_error', None):
            raise RuntimeError('Serial reception failed: ' + self.board.recv_error)
        updated = False
        while True:
            item = self.board.get_wheel_feedback()
            if item is None:
                break
            sample, received_ns = item
            if self.resets != self.board.wheel_feedback_tracker.resets:
                self.window = Window(minimum_ms=20)
                self.rest.reset()
                self.measurement = None
                self.measurement_at = None
                self.resets = self.board.wheel_feedback_tracker.resets
            measurement = self.window.update(sample)
            if measurement is not None:
                self.measurement = measurement
                self.measurement_at = received_ns
            resting, duration, gap = self.rest.update(sample)
            self.latest = dict(asdict(sample), received_monotonic_ns=received_ns,
                               wheel_names=WHEEL_NAMES,
                               measurement_valid=measurement is not None,
                               measured_rps=None if measurement is None else measurement[0],
                               measured_window_ms=None if measurement is None else measurement[1],
                               encoder_rest=resting, rest_duration_ms=duration,
                               rest_max_gap_ms=gap)
            updated = True
        return updated

    def snapshot(self):
        transport = dict(rx_bytes=getattr(self.board, 'rx_bytes', 0),
                         valid_frames=getattr(self.board, 'valid_frames', 0),
                         crc_errors=getattr(self.board, 'crc_errors', 0),
                         bad_frames=self.board.wheel_feedback_bad_frames)
        if self.latest is None:
            return dict(status='WAITING_FOR_FEEDBACK', fresh=False, **transport)
        result = self.latest.copy()
        result.update(transport)
        measurement_fresh = (self.measurement_at is not None and
                             0 <= (time.monotonic_ns() - self.measurement_at) / 1e9 <= .2)
        result.update(measurement_valid=measurement_fresh,
                      measured_rps=self.measurement[0] if measurement_fresh else None,
                      measured_window_ms=self.measurement[1] if measurement_fresh else None)
        age = (time.monotonic_ns() - result['received_monotonic_ns']) / 1e9
        result.update(age_s=age, fresh=0 <= age <= 0.2,
                      received_gap_count=self.board.wheel_feedback_tracker.lost,
                      mcu_resets=self.board.wheel_feedback_tracker.resets,
                      bad_frames=self.board.wheel_feedback_bad_frames,
                      host_queue_drops=self.board.wheel_feedback_queue_drops)
        result['status'] = 'FEEDBACK' if result['fresh'] else 'STALE_FEEDBACK'
        if not result['fresh']:
            result.update(measurement_valid=False, measured_rps=None, encoder_rest=False)
        result['body_velocity'] = None
        if self.geometry is not None:
            result['geometry_m'] = asdict(self.geometry)
            if result['measurement_valid']:
                vx, vy, wz = self.geometry.to_body_velocity(result['measured_rps'])
                result['body_velocity'] = dict(vx_m_s=vx, vy_m_s=vy, wz_rad_s=wz)
        return result


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/rrc')
    parser.add_argument('--baudrate', type=int, default=1000000)
    commands = parser.add_subparsers(dest='mode', required=True)
    feedback = commands.add_parser('feedback', help='Read only; sends no motor commands')
    feedback.add_argument('--duration', type=float, default=10.0)
    def dimensions(command, required=False):
        for name in ('wheel-diameter', 'wheelbase', 'track-width'):
            command.add_argument('--' + name, type=float, required=required,
                                 help='Chassis dimension in meters (explicit geometry)')
    dimensions(feedback)
    drive = commands.add_parser('drive', help='Four raw motor rps, refreshed at 20 Hz')
    drive.add_argument('--rps', type=float, nargs=4, required=True,
                       metavar=('LF', 'LR', 'RF', 'RR'))
    drive.add_argument('--duration', type=float, default=2.0)
    drive.add_argument('--max-rps', type=float, default=0.2,
                       help='Absolute per-wheel limit (default 0.2 rps = 12 rpm)')
    dimensions(drive)
    velocity = commands.add_parser('velocity', help='Mecanum body vx/vy/wz -> four wheel rps')
    velocity.add_argument('--vx', type=float, required=True, help='Forward m/s')
    velocity.add_argument('--vy', type=float, default=0.0, help='Leftward m/s')
    velocity.add_argument('--wz', type=float, default=0.0, help='Counterclockwise rad/s')
    velocity.add_argument('--duration', type=float, default=2.0)
    velocity.add_argument('--max-rps', type=float, default=0.2)
    dimensions(velocity, required=True)
    commands.add_parser('stop', help='Send zero to all four wheels')
    args = parser.parse_args(argv)
    if args.baudrate <= 0:
        parser.error('baudrate must be positive')
    if args.mode != 'stop' and (not math.isfinite(args.duration) or not 0 < args.duration <= 60):
        parser.error('duration must be finite, greater than zero, at most 60 seconds')
    args.geometry = None
    if args.mode != 'stop':
        values = [args.wheel_diameter, args.wheelbase, args.track_width]
        if any(v is not None for v in values):
            if any(v is None for v in values):
                parser.error('Provide all three geometry dimensions together')
            try:
                args.geometry = MecanumKinematics(*values)
            except ValueError as exc:
                parser.error(str(exc))
    if args.mode == 'velocity':
        try:
            args.rps = args.geometry.to_wheel_rps(args.vx, args.vy, args.wz)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            parser.error(str(exc))
    if args.mode in ('drive', 'velocity'):
        if not math.isfinite(args.max_rps) or args.max_rps <= 0:
            parser.error('max-rps must be positive and finite')
        if any(not math.isfinite(v) or abs(v) > args.max_rps for v in args.rps):
            parser.error('every rps must be finite and within max-rps')
    return args


def interrupt(signum, frame):
    raise KeyboardInterrupt


def run(args, board):
    if args.mode == 'stop':
        board.set_motor_speed(ZERO)
        return
    board.enable_reception(True)
    reader = EncoderReader(board, args.geometry)
    if args.mode in ('drive', 'velocity'):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            reader.poll()
            if reader.snapshot()['fresh']:
                break
            time.sleep(0.002)
        else:
            raise RuntimeError('No fresh 0x70 encoder feedback; refusing nonzero commands')
    end = time.monotonic() + args.duration
    next_command = next_print = 0.0
    while time.monotonic() < end:
        reader.poll()
        state = reader.snapshot()
        now = time.monotonic()
        if args.mode in ('drive', 'velocity'):
            if not state['fresh']:
                raise RuntimeError('Encoder feedback older than 200 ms; stopping all wheels')
            if state['mcu_resets'] != reader.resets or state['mcu_resets']:
                raise RuntimeError('MCU reset detected; stopping all wheels')
            if not state['flags'] & 1:
                raise RuntimeError('MCU safety gate inhibits motors; stopping all wheels')
            if now >= next_command:
                board.set_motor_speed(list(enumerate(args.rps, start=1)))
                next_command = now + 0.05
        if now >= next_print:
            # Reprint stale state explicitly; never turn absent feedback into zero speed.
            print(json.dumps(state, allow_nan=False), flush=True)
            next_print = now + 0.1
        time.sleep(0.002)


def main(argv=None):
    args = options(argv)
    signal.signal(signal.SIGTERM, interrupt)
    board = None
    failed = False
    try:
        board = Board(args.device, args.baudrate, timeout=0.05, exclusive=True)
        run(args, board)
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        failed = True
    finally:
        if board is not None:
            try:
                if args.mode in ('drive', 'velocity', 'stop'):
                    for _ in range(3):
                        board.set_motor_speed(ZERO)
                        time.sleep(0.02)
            except Exception as exc:
                print('Unable to send stop: ' + str(exc), file=sys.stderr)
                failed = True
            finally:
                board.close()
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
