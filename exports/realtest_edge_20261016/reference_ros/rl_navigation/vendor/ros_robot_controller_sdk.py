#!/usr/bin/env python3
# encoding: utf-8
# stm32 python sdk
import enum
import time
import queue
import struct
import serial
import threading

from .wheel_feedback import decode, Tracker

class PacketControllerState(enum.IntEnum):
    # 通信协议的格式(the format of the communication protocol)
    # 0xAA 0x55 Length Function ID Data Checksum
    PACKET_CONTROLLER_STATE_STARTBYTE1 = 0
    PACKET_CONTROLLER_STATE_STARTBYTE2 = 1
    PACKET_CONTROLLER_STATE_LENGTH = 2
    PACKET_CONTROLLER_STATE_FUNCTION = 3
    PACKET_CONTROLLER_STATE_ID = 4
    PACKET_CONTROLLER_STATE_DATA = 5
    PACKET_CONTROLLER_STATE_CHECKSUM = 6

class PacketFunction(enum.IntEnum):
    # 可通过串口实现的控制功能(achieve control function via the serial port)
    PACKET_FUNC_SYS = 0
    PACKET_FUNC_LED = 1  # LED控制(LED control)
    PACKET_FUNC_BUZZER = 2  # 蜂鸣器控制(buzzer control)
    PACKET_FUNC_MOTOR = 3  # 电机控制(motor control)
    PACKET_FUNC_PWM_SERVO = 4  # PWM舵机控制, 板子上从里到外依次为1-4(PWM servo control. The servos on the board are numbered 1 to 4 from inside to outside)
    PACKET_FUNC_BUS_SERVO = 5  # 总线舵机控制(bus servo control)
    PACKET_FUNC_KEY = 6  # 按键获取(obtain button)
    PACKET_FUNC_IMU = 7  # IMU获取(obtain IMU)
    PACKET_FUNC_GAMEPAD = 8  # 手柄获取(obtain handle)
    PACKET_FUNC_SBUS = 9  # 航模遥控获取(obtain model aircraft remote control)
    PACKET_FUNC_OLED = 10 # OLED 显示内容设置(set OLED display content)
    PACKET_FUNC_ULTRASONIC = 11
    PACKET_FUNC_RGB = 12
    PACKET_FUNC_NONE = 13

class PacketReportKeyEvents(enum.IntEnum):
    # 按键的不同状态(different button status)
    KEY_EVENT_PRESSED = 0x01
    KEY_EVENT_LONGPRESS = 0x02
    KEY_EVENT_LONGPRESS_REPEAT = 0x04
    KEY_EVENT_RELEASE_FROM_LP = 0x08
    KEY_EVENT_RELEASE_FROM_SP = 0x10
    KEY_EVENT_CLICK = 0x20
    KEY_EVENT_DOUBLE_CLICK= 0x40
    KEY_EVENT_TRIPLE_CLICK = 0x80

crc8_table = [
    0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
    157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
    35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
    190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
    70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
    219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
    248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
    140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
    17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
    202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
    87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
    233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
    116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53
]

def checksum_crc8(data):
    # 校验(check)
    check = 0
    for b in data:
        check = crc8_table[check ^ b]
    return check & 0x00FF

class SBusStatus:
    def __init__(self):
        self.channels = [0] * 16
        self.channel_17 = False
        self.channel_18 = False
        self.signal_loss = True
        self.fail_safe = False

class Board:
    buttons_map = {
            'GAMEPAD_BUTTON_MASK_L2':        0x0001,
            'GAMEPAD_BUTTON_MASK_R2':        0x0002,
            'GAMEPAD_BUTTON_MASK_SELECT':    0x0004,
            'GAMEPAD_BUTTON_MASK_START':     0x0008,
            'GAMEPAD_BUTTON_MASK_MODE':      0x0010,
            'GAMEPAD_BUTTON_MASK_L3':        0x0020,
            'GAMEPAD_BUTTON_MASK_R3':        0x0040,
            'GAMEPAD_BUTTON_MASK_CROSS':     0x0100,
            'GAMEPAD_BUTTON_MASK_CIRCLE':    0x0200,
            'GAMEPAD_BUTTON_MASK_SQUARE':    0x0800,
            'GAMEPAD_BUTTON_MASK_TRIANGLE':  0x1000,
            'GAMEPAD_BUTTON_MASK_L1':        0x4000,
            'GAMEPAD_BUTTON_MASK_R1':        0x8000
    }

    def __init__(self, device="/dev/rrc", baudrate=1000000, timeout=5, exclusive=False):
        self.enable_recv = False
        self._recv_stop = threading.Event()
        self.recv_error = None
        self.rx_bytes = 0
        self.valid_frames = 0
        self.crc_errors = 0
        self.frame = []
        self.recv_count = 0

        self.port = serial.Serial(None, baudrate, timeout=timeout)
        self.port.write_timeout = 0.1
        self.port.exclusive = exclusive
        self.port.rts = False
        self.port.dtr = False
        self.port.setPort(device)
        self.port.open()

        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1
        self.servo_read_lock = threading.Lock()
        self.pwm_servo_read_lock = threading.Lock()
        self.motor_status_lock = threading.Lock()

        # 队列用来存储数据(use queue to store data)
        self.wheel_feedback_queue = queue.Queue(maxsize=32)
        self.wheel_feedback_tracker = Tracker()
        self.wheel_feedback_bad_frames = 0
        self.wheel_feedback_queue_drops = 0
        self.motor_status_queue = queue.Queue(maxsize=1)
        self.sys_queue = queue.Queue(maxsize=1)
        self.bus_servo_queue = queue.Queue(maxsize=1)
        self.pwm_servo_queue = queue.Queue(maxsize=1)
        self.key_queue = queue.Queue(maxsize=1)
        self.imu_queue = queue.Queue(maxsize=1)
        self.ultrasonic_queue = queue.Queue(maxsize=1)
        self.gamepad_queue = queue.Queue(maxsize=1)
        self.sbus_queue = queue.Queue(maxsize=1)

        self.parsers = {
            PacketFunction.PACKET_FUNC_MOTOR: self.packet_report_motor,
            PacketFunction.PACKET_FUNC_SYS: self.packet_report_sys,
            PacketFunction.PACKET_FUNC_KEY: self.packet_report_key,
            PacketFunction.PACKET_FUNC_IMU: self.packet_report_imu,
            PacketFunction.PACKET_FUNC_ULTRASONIC: self.packet_report_ultrasonic,
            PacketFunction.PACKET_FUNC_GAMEPAD: self.packet_report_gamepad,
            PacketFunction.PACKET_FUNC_BUS_SERVO: self.packet_report_serial_servo,
            PacketFunction.PACKET_FUNC_SBUS: self.packet_report_sbus,
            PacketFunction.PACKET_FUNC_PWM_SERVO: self.packet_report_pwm_servo
        }

        time.sleep(0.5)
        self._recv_thread = threading.Thread(target=self._recv_runner, daemon=True)
        self._recv_thread.start()


    def packet_report_wheel_feedback(self, data):
        if not data or data[0] != 0x70:
            return
        try:
            sample = decode(data)
        except ValueError:
            self.wheel_feedback_bad_frames += 1
            return
        if not self.wheel_feedback_tracker.accept(sample):
            return
        item = (sample, time.monotonic_ns())
        try:
            self.wheel_feedback_queue.put_nowait(item)
        except queue.Full:
            # The publishing thread may have drained the queue after Full.
            try:
                self.wheel_feedback_queue.get_nowait()
            except queue.Empty:
                pass
            self.wheel_feedback_queue_drops += 1
            self.wheel_feedback_queue.put_nowait(item)

    def get_wheel_feedback(self):
        try:
            return self.wheel_feedback_queue.get_nowait()
        except queue.Empty:
            return None

    def packet_report_motor(self, data):
        if data and data[0] == 0x70:
            self.packet_report_wheel_feedback(data)
            return
        if len(data) != 19 or bytes(data[:2]) != bytes((6, 1)):
            return
        try:
            self.motor_status_queue.put_nowait(bytes(data))
        except queue.Full:
            pass

    def get_motor_status(self, timeout=0.3):
        """Read MCU interlocks/counters only; does not enable or command motors."""
        if not self.enable_recv:
            return None
        with self.motor_status_lock:
            try:
                self.motor_status_queue.get_nowait()
            except queue.Empty:
                pass
            self.buf_write(PacketFunction.PACKET_FUNC_MOTOR, bytes((6,)))
            try:
                data = self.motor_status_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        values = struct.unpack('<7B3I', data)
        result = dict(zip(('k1', 'k2', 'pd3', 'enabled', 'reason',
                           'control_ticks', 'target_commands', 'rejected_commands'), values[2:]))
        reasons = ('ready', 'estop_contacts', 'pd3_inhibit', 'release_delay')
        result['reason'] = reasons[result['reason']] if result['reason'] < len(reasons) else 'unknown'
        return result

    def packet_report_sys(self, data):
        try:
            self.sys_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_key(self, data):
        try:
            self.key_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_imu(self, data):
        if len(data) != 24:
            return
        sample = (data, time.monotonic_ns())
        try:
            self.imu_queue.put_nowait(sample)
        except queue.Full:
            try:
                self.imu_queue.get_nowait()
            except queue.Empty:
                pass
            self.imu_queue.put_nowait(sample)

    def get_imu_sample(self):
        if not self.enable_recv:
            return None
        try:
            data, received_ns = self.imu_queue.get_nowait()
            return struct.unpack('<6f', data), received_ns
        except queue.Empty:
            return None

    def packet_report_gamepad(self, data):
        try:
            self.gamepad_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_serial_servo(self, data):
        try:
            self.bus_servo_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_pwm_servo(self, data):
        try:
            self.pwm_servo_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_sbus(self, data):
        try:
            self.sbus_queue.put_nowait(data)
        except queue.Full:
            pass

    def get_battery(self):
        # 获取电压，单位mAh(obtain voltage, which is in the unit of mAh)
        if self.enable_recv:
            try:
                data = self.sys_queue.get(block=False)
                if data[0] == 0x04:
                    return struct.unpack('<H', data[1:])[0]
                else:
                    None
            except queue.Empty:
                return None
        else:
            print('get_battery enable reception first!')
            return None

    def get_button(self):
        # 获取按键key1， key2状态，返回按键ID(1表示按键1，2表示按键2)和状态(0表示按下，1表示松开)(Get the status of buttons 1 and 2, and return the button ID (1 for key1, and 2 for key2) and status (0 for pressed, and 1 for released))
        if self.enable_recv:
            try:
                data = self.key_queue.get(block=False)
                key_id = data[0]
                key_event = PacketReportKeyEvents(data[1])
                if key_event == PacketReportKeyEvents.KEY_EVENT_CLICK:
                    return key_id, 0
                elif key_event == PacketReportKeyEvents.KEY_EVENT_PRESSED:
                    return key_id, 1
            except queue.Empty:
                return None
        else:
            print('get_button enable reception first!')
            return None

    def get_imu(self):
        sample = self.get_imu_sample()
        return None if sample is None else sample[0]

    def packet_report_ultrasonic(self, data):
        if len(data) != 12:
            return
        # Keep only the latest complete frame, with its receive time.
        try:
            self.ultrasonic_queue.get_nowait()
        except queue.Empty:
            pass
        self.ultrasonic_queue.put_nowait((time.monotonic(), bytes(data)))

    def get_ultrasonic(self):
        """Return [(id, distance_mm)] for IDs 1..4; 65535 means invalid.

        CH1245 firmware: P11, P12, P14, P15. None means no fresh frame."""
        if not self.enable_recv:
            return None
        try:
            received, data = self.ultrasonic_queue.get_nowait()
        except queue.Empty:
            return None
        if time.monotonic() - received > 1.0:
            return None
        return [(sensor_id, distance if success == 1 else 65535)
                for sensor_id, (success, distance) in
                enumerate(struct.iter_unpack('<BH', data), start=1)]

    def set_rgb_segments(self, counts):
        """Configure 1..8 contiguous sections, total 1..20 pixels; clears LEDs."""
        counts = tuple(counts)
        if not 1 <= len(counts) <= 8 or any(type(v) is not int or not 1 <= v <= 20 for v in counts) or sum(counts) > 20:
            raise ValueError('counts: 1..8 positive integers, total <=20')
        data = struct.pack('<4B' + 'H' * len(counts), 0x10, 1, len(counts), 0, *counts)
        self.buf_write(PacketFunction.PACKET_FUNC_RGB, data)
        self.rgb_segments = counts

    def set_rgb_effect(self, segment=0, mode=1, rgb=(255, 255, 255), brightness=128, period_ms=3000, repeat=0):
        """0=whole strip, 1..N=section; repeat 0=infinite, modes 2..6 stop after N cycles.

        A whole effect replaces all sections. A section replaces a whole effect
        while preserving a uniform solid background; other whole effects clear the remaining sections.
        """
        counts = getattr(self, 'rgb_segments', (2, 2, 2, 2))
        if type(segment) is not int or not 0 <= segment <= len(counts):
            raise ValueError('segment must be 0..configured section count')
        rgb = tuple(rgb)
        if type(mode) is not int or not 0 <= mode <= 6:
            raise ValueError('mode must be 0..6')
        if len(rgb) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in (*rgb, brightness)):
            raise ValueError('RGB and brightness must be integers in 0..255')
        if type(period_ms) is not int or not 0 <= period_ms <= 65535 or (mode >= 2 and not 200 <= period_ms <= 60000):
            raise ValueError('period_ms: uint16; effects require 200..60000 ms')
        if type(repeat) is not int or not 0 <= repeat <= 65535:
            raise ValueError('repeat must be 0..65535')
        self.buf_write(PacketFunction.PACKET_FUNC_RGB,
                       struct.pack('<7BHH', 0x11, segment, mode, *rgb, brightness, period_ms, repeat))

    def set_rgb(self, count=10, mode=1, rgb=(255, 255, 255), brightness=128, period_ms=3000):
        """Strip: 1..20 pixels. Mode: 0 off, 1 solid, 2 breath, 3 flash, 4 chase, 5 center expand, 6 RGB chase.

        Mode 6 moves RGB by configured sections when count equals their total, otherwise by pixel.
        RGB/brightness: 0..255; effect period: 200..60000 ms. No device ACK.
        """
        if not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError('count must be 1..20')
        if not isinstance(mode, int) or mode not in range(7):
            raise ValueError('mode must be 0..6')
        rgb = tuple(rgb)
        if len(rgb) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in (*rgb, brightness)):
            raise ValueError('RGB and brightness must be integers in 0..255')
        if not isinstance(period_ms, int) or not 0 <= period_ms <= 65535:
            raise ValueError('period_ms must fit uint16')
        if mode >= 2 and not 200 <= period_ms <= 60000:
            raise ValueError('effect period_ms must be 200..60000')
        self.buf_write(PacketFunction.PACKET_FUNC_RGB,
                       struct.pack('<H5BH', count, mode, *rgb, brightness, period_ms))

    def set_ultrasonic_led(self, sensor_id, mode, left_rgb, right_rgb=None, period_ms=1000):
        """ID: 0=all, 1..4=P11/P12/P14/P15 (CH1245 firmware). Mode: 0=solid, 1=breath, 2=flash, 3=swap.

        RGB: 0..255 per color. period_ms: full cycle, 200..60000 for effects.
        """
        if sensor_id not in range(5) or mode not in range(4):
            raise ValueError('sensor_id must be 0..4; mode must be 0..3')
        right_rgb = left_rgb if right_rgb is None else right_rgb
        rgb = tuple(left_rgb) + tuple(right_rgb)
        if len(left_rgb) != 3 or len(right_rgb) != 3 or any(v not in range(256) for v in rgb):
            raise ValueError('each RGB must contain three integers in 0..255')
        period_ms = 0 if mode == 0 else period_ms
        if mode != 0 and (not isinstance(period_ms, int) or not 200 <= period_ms <= 60000):
            raise ValueError('period_ms must be an integer in 200..60000')
        data = struct.pack('<8BH', sensor_id, mode, *rgb, period_ms)
        self.buf_write(PacketFunction.PACKET_FUNC_ULTRASONIC, data)

    def get_gamepad(self):
        # 获取手柄数据(obtain handle data)
        if self.enable_recv:
            try:
                # buttons, hat, lx, ly, rx, ry
                gamepad_data = struct.unpack("<HB4b", self.gamepad_queue.get(block=False))
                # 'lx', 'ly', 'rx', 'ry', 'r2', 'l2', 'hat_x', 'hat_y'
                axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # 'cross', 'circle', '', 'square', 'triangle', '', 'l1', 'r1', 'l2', 'r2', 'select', 'start', 'mode', 'l3', 'r3', ''
                buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
                for b in self.buttons_map:
                    if self.buttons_map[b] & gamepad_data[0]:
                        if b == 'GAMEPAD_BUTTON_MASK_R2':
                            axes[4] = 1.0
                        elif b == 'GAMEPAD_BUTTON_MASK_L2':
                            axes[5] = 1.0
                        elif b == 'GAMEPAD_BUTTON_MASK_CROSS':
                            buttons[0] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_CIRCLE':
                            buttons[1] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_SQUARE':
                            buttons[3] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_TRIANGLE':
                            buttons[4] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_L1':
                            buttons[6] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_R1':
                            buttons[7] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_SELECT':
                            buttons[10] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_START':
                            buttons[11] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_MODE':
                            buttons[12] = 1
               
                if gamepad_data[2] > 0:
                    axes[0] = -gamepad_data[2] / 127
                elif gamepad_data[2] < 0:
                    axes[0] = -gamepad_data[2] / 128

                if gamepad_data[3] > 0:
                    axes[1] = gamepad_data[3] / 127
                elif gamepad_data[3] < 0:
                    axes[1] = gamepad_data[3] / 128

                if gamepad_data[4] > 0:
                    axes[2] = -gamepad_data[4] / 127
                elif gamepad_data[4] < 0:
                    axes[2] = -gamepad_data[4] / 128

                if gamepad_data[5] > 0:
                    axes[3] = gamepad_data[5] / 127
                elif gamepad_data[5] < 0:
                    axes[3] = gamepad_data[5] / 128
            
                if gamepad_data[1] == 9:
                    axes[6] = 1.0
                elif gamepad_data[1] == 13:
                    axes[6] = -1.0
                
                if gamepad_data[1] == 11:
                    axes[7] = -1.0
                elif gamepad_data[1] == 15:
                    axes[7] = 1.0
                return axes, buttons
            except queue.Empty:
                return None
        else:
            print('get_gamepad enable reception first!')
            return None

    def get_sbus(self):
        if self.enable_recv:
            try:
                sbus_data = self.sbus_queue.get(block=False)
                status = SBusStatus()
                *status.channels, ch17, ch18, sig_loss, fail_safe = struct.unpack("<16hBBBB", sbus_data)
                status.channel_17 = ch17 != 0
                status.channel_18 = ch18 != 0
                status.signal_loss = sig_loss != 0
                status.fail_safe = fail_safe != 0
                data = []
                if status.signal_loss:
                    data = 16 * [0.5]
                    data[4] = 0
                    data[5] = 0
                    data[6] = 0
                    data[7] = 0
                else:
                    for i in status.channels:
                        data.append(2*(i - 192)/(1792 - 192) - 1)
                return data
            except queue.Empty:
                return None
        else:
            print('get_sbus enable reception first!')
            return None

    def buf_write(self, func, data):
        buf = [0xAA, 0x55, int(func)]
        buf.append(len(data))
        buf.extend(data)
        buf.append(checksum_crc8(bytes(buf[2:])))
        self.port.write(buf)

    def set_led(self, on_time, off_time, repeat=1, led_id=1):
        on_time = int(on_time*1000)
        off_time = int(off_time*1000)
        self.buf_write(PacketFunction.PACKET_FUNC_LED, struct.pack("<BHHH", led_id, on_time, off_time, repeat))

    def set_buzzer(self, freq, on_time, off_time, repeat=1):
        on_time = int(on_time*1000)
        off_time = int(off_time*1000)
        self.buf_write(PacketFunction.PACKET_FUNC_BUZZER, struct.pack("<HHHH", freq, on_time, off_time, repeat))

    def set_motor_speed(self, speeds):
        data = [0x01, len(speeds)]
        for i in speeds:
            data.extend(struct.pack("<Bf", int(i[0] - 1), float(i[1])))
        self.buf_write(PacketFunction.PACKET_FUNC_MOTOR, data)

    def set_oled_text(self, line, text):
        data = [line, len(text)] # 子命令为 0x01 设置 SSID, 第二个字节是字符串长度，该长度包含'\0'字符串结束符(The sub-command 0x01 is used to set the SSID. The second byte is the length of the string, which includes the '\0' string termination character)
        data.extend(bytes(text, encoding='utf-8'))
        self.buf_write(PacketFunction.PACKET_FUNC_OLED, data)

    def pwm_servo_set_position(self, duration, positions):
        duration = int(duration * 1000)
        data = [0x03, duration & 0xFF, 0xFF & (duration >> 8)]
        # data = [0x01, duration & 0xFF, 0xFF & (duration >> 8), len(positions)]
        for i in positions:
            data.extend(struct.pack("<BH", i[0], i[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_PWM_SERVO, data)
    
    def pwm_servo_set_offset(self, servo_id, offset):
        data = struct.pack("<BBb", 0x07, servo_id, int(offset))
        self.buf_write(PacketFunction.PACKET_FUNC_PWM_SERVO, data)

    def pwm_servo_read_and_unpack(self, servo_id, cmd, unpack):
        with self.servo_read_lock:
            self.buf_write(PacketFunction.PACKET_FUNC_PWM_SERVO, [cmd, servo_id])
            data = self.pwm_servo_queue.get(block=True)
            servo_id, cmd, info = struct.unpack(unpack, data)
            return info

    def pwm_servo_read_offset(self, servo_id):
        return self.pwm_servo_read_and_unpack(servo_id, 0x09, "<BBb")

    def pwm_servo_read_position(self, servo_id):
        return self.pwm_servo_read_and_unpack(servo_id, 0x05, "<BBH")

    def bus_servo_enable_torque(self, servo_id, enable):
        if enable:
            data = struct.pack("<BB", 0x0B, servo_id)
        else:
            data = struct.pack("<BB", 0x0C, servo_id)
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_id(self, servo_id_now, servo_id_new):
        data = struct.pack("<BBB", 0x10, servo_id_now, servo_id_new)
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_offset(self, servo_id, offset):
        data = struct.pack("<BBb", 0x20, servo_id, int(offset))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_save_offset(self, servo_id):
        data = struct.pack("<BB", 0x24, servo_id)
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_angle_limit(self, servo_id, limit):
        data = struct.pack("<BBHH", 0x30, servo_id, int(limit[0]), int(limit[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_vin_limit(self, servo_id, limit):
        data = struct.pack("<BBHH", 0x34, servo_id, int(limit[0]), int(limit[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_temp_limit(self, servo_id, limit):
        data = struct.pack("<BBb", 0x38, servo_id, int(limit))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_stop(self, servo_id):
        data = [0x03, len(servo_id)] 
        data.extend(struct.pack("<"+'B'*len(servo_id), *servo_id))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)

    def bus_servo_set_position(self, duration, positions):
        duration = int(duration * 1000)
        data = [0x01, duration & 0xFF, 0xFF & (duration >> 8), len(positions)] # 0x00 is bus servo sub command
        for i in positions:
            data.extend(struct.pack("<BH", i[0], i[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)

    def bus_servo_read_and_unpack(self, servo_id, cmd, unpack):
        with self.servo_read_lock:
            self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, [cmd, servo_id])
            data = self.bus_servo_queue.get(block=True)
            servo_id, cmd, success, *info = struct.unpack(unpack, data)
            if success == 0:
                return info

    def bus_servo_read_id(self, servo_id=254):
        return self.bus_servo_read_and_unpack(servo_id, 0x12, "<BBbB")

    def bus_servo_read_offset(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x22, "<BBbb")
    
    def bus_servo_read_position(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x05, "<BBbh")

    def bus_servo_read_vin(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x07, "<BBbH")
    
    def bus_servo_read_temp(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x09, "<BBbB")

    def bus_servo_read_temp_limit(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x3A, "<BBbB")

    def bus_servo_read_angle_limit(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x32, "<BBb2H")

    def bus_servo_read_vin_limit(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x36, "<BBb2H")

    def bus_servo_read_torque_state(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x0D, "<BBbb")
       
     
    def hx_bus_servo_read_id(self, servo_id=254):
        with self.servo_read_lock:
            self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, [0x40, 0x00])
            try:
                data = self.bus_servo_queue.get(block=True, timeout=10)
            except queue.Empty:
                return None
            if len(data) < 3:
                return []
            count = data[2]
            ids = list(data[3:3 + count])
            return ids

    def hx_bus_servo_read_pid(self, servo_id):

        with self.servo_read_lock:
            self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, [0x42, servo_id])
            try:
                data = self.bus_servo_queue.get(block=True, timeout=10)
            except queue.Empty:
                return None
            
            if len(data) < 8:
                return None
            servo_id_r, cmd_r, success, p, d, i, force = struct.unpack("<BBbBBBH", data[:8])
            if success == 0:
                return (p, d, i, force)
            return None

    def hx_bus_servo_read_min_start_force(self, servo_id):
        result = self.hx_bus_servo_read_pid(servo_id)
        if result is not None:
            return (result[3],)
        return None

    def hx_bus_servo_set_pid(self, servo_id, p, d, i, min_start_force=0):
        data = struct.pack("<BBBBBH", 0x43, int(servo_id) & 0xFF,
                           int(p) & 0xFF, int(d) & 0xFF, int(i) & 0xFF, int(min_start_force) & 0xFFFF)
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def hx_bus_servo_set_min_start_force(self, servo_id, force):
        current = self.hx_bus_servo_read_pid(servo_id)
        if current is not None:
            p, d, i, _ = current
            self.hx_bus_servo_set_pid(servo_id, p, d, i, force)

    def hx_bus_servo_set_baudrate(self, servo_id, old_baud_index=0, new_baud_index=4):
        data = struct.pack("<BBBB", 0x44, servo_id, int(old_baud_index), int(new_baud_index))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.1)

    def hx_bus_servo_set_all_baudrate(self, old_baud_index=0, new_baud_index=4):
        self.hx_bus_servo_set_baudrate(0xFE, old_baud_index, new_baud_index)
        
    def set_motor_type(self, motor_type=0x02):
        '''
        MOTOR_TYPE_JGB520 0x00
        MOTOR_TYPE_JGB37  0x01
        MOTOR_TYPE_JGB27  0x02
        MOTOR_TYPE_JGB528 0x03
        '''
        self.buf_write(PacketFunction.PACKET_FUNC_MOTOR, struct.pack("<BB",0x05,motor_type))


    def set_battery_level(self,battery_level=0x2904):
        '''
        battery_level 
        6V     0x1770
        6.9V   0x1af4 (Mecanum,Acker) 
        7V     0x1b58
        8V     0x1f40
        9V     0x2328
        10V    0x2710
        10.5V  0x2904 
        11V    0x2af8 (Tank)
        12V    0x2ee0
        '''
        high_byte = (battery_level >> 8) & 0xFF
        low_byte = battery_level & 0xFF
        self.buf_write(PacketFunction.PACKET_FUNC_SYS, struct.pack("BBB", 0x01, low_byte,high_byte))
        
    def enable_reception(self, enable=True):
        self.enable_recv = enable

    def _recv_runner(self):
        try:
            self.recv_task()
        except (serial.SerialException, OSError) as exc:
            self.recv_error = str(exc)

    def close(self):
        """Close reception; caller must send any required motor stop first."""
        self.enable_recv = False
        self._recv_stop.set()
        self._recv_thread.join(timeout=self.port.timeout + 0.5)
        if self.port.is_open:
            self.port.close()

    def recv_task(self):
        while not self._recv_stop.is_set():
            if self.enable_recv:
                recv_data = self.port.read()
                if recv_data:
                    self.rx_bytes = getattr(self, 'rx_bytes', 0) + len(recv_data)
                    for dat in recv_data:
                        # print("%0.2X "%dat)
                        if self.state == PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1:
                            if dat == 0xAA:
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE2
                            continue
                        elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE2:
                            if dat == 0x55:
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_FUNCTION
                            else:
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1
                            continue
                        elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_FUNCTION:
                            if dat < int(PacketFunction.PACKET_FUNC_NONE):
                                self.frame = [dat, 0]
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_LENGTH
                            else:
                                self.frame = []
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1
                            continue
                        elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_LENGTH:
                            self.frame[1] = dat
                            self.recv_count = 0
                            if dat == 0:
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_CHECKSUM
                            else:
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_DATA
                            continue
                        elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_DATA:
                            self.frame.append(dat)
                            self.recv_count += 1
                            if self.recv_count >= self.frame[1]:
                                self.state = PacketControllerState.PACKET_CONTROLLER_STATE_CHECKSUM
                            continue
                        elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_CHECKSUM:
                            crc8 = checksum_crc8(bytes(self.frame))
                            if crc8 == dat:
                                self.valid_frames = getattr(self, 'valid_frames', 0) + 1
                                func = PacketFunction(self.frame[0])
                                data = bytes(self.frame[2:])
                                if func in self.parsers:
                                    self.parsers[func](data)
                            else:
                                self.crc_errors = getattr(self, 'crc_errors', 0) + 1
                                print("校验失败")
                            self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1
                            continue
            else:
                time.sleep(0.01)
        self.port.close()
        print("END...")
def hx_servo_test(board):
    # 第一步: 改波特率 (只需执行一次, 改完后注释掉这两行)
    print("正在修改舵机波特率为 115200 ...")
    board.hx_bus_servo_set_all_baudrate(old_baud_index=0, new_baud_index=4)
    time.sleep(0.5)
    print("波特率修改完成")

    # 第二步: 扫描所有在线舵机 (STM32 端自动扫描 1~253)
    print("正在扫描舵机 ...")
    online_ids = board.hx_bus_servo_read_id()
    print("在线舵机 ID:", online_ids)

    if online_ids:
        # 第三步: 读取每个舵机的 PID 和最小启动力
        for sid in online_ids:
            result = board.hx_bus_servo_read_pid(sid)
            if result:
                p, d, i, force = result
                print(f"  ID={sid}: P={p}, D={d}, I={i}, 最小启动力={force}")

        # 第四步: 设置 PID 和最小启动力 ()
        for sid in online_ids:
            board.hx_bus_servo_set_pid(sid, p=48, d=48, i=0, min_start_force=16)
        print("设置完成")
def bus_servo_test(board):
    board.bus_servo_set_position(1, [[1, 500], [2, 500]])
    time.sleep(1)
    board.bus_servo_set_position(2, [[1, 0], [2, 0]])
    time.sleep(1)
    board.bus_servo_stop([1, 2])
    time.sleep(1)
    
    servo_id = 1
    board.bus_servo_set_id(254, servo_id)
    servo_id = board.bus_servo_read_id()
    if servo_id is not None:
        servo_id = servo_id[0]
        
        offset_set = -10
        board.bus_servo_set_offset(servo_id, offset_set)
        board.bus_servo_save_offset(servo_id)
        
        vin_l, vin_h = 4500, 14500
        board.bus_servo_set_vin_limit(servo_id, [vin_l, vin_h])

        temp_limit = 85
        board.bus_servo_set_temp_limit(servo_id, temp_limit)

        angle_l, angle_h = 0, 1000
        board.bus_servo_set_angle_limit(servo_id, [angle_l, angle_h])
        
        board.bus_servo_enable_torque(servo_id, 1)

        print('id:', board.bus_servo_read_id(servo_id))
        print('offset:', board.bus_servo_read_offset(servo_id), offset_set)
        print('vin:', board.bus_servo_read_vin(servo_id))
        print('temp:', board.bus_servo_read_temp(servo_id))
        print('position:', board.bus_servo_read_position(servo_id))
        print('angle_limit:', board.bus_servo_read_angle_limit(servo_id), [angle_l, angle_h])
        print('vin_limit:', board.bus_servo_read_vin_limit(servo_id), [vin_l, vin_h])
        print('temp_limit:', board.bus_servo_read_temp_limit(servo_id), temp_limit)
        print('torque_state:', board.bus_servo_read_torque_state(servo_id))

def pwm_servo_test(board):
    servo_id = 1
    board.pwm_servo_set_position(0.5, [[servo_id, 1500]])
    board.pwm_servo_set_offset(servo_id, 0)
    print('offset:', board.pwm_servo_read_offset(servo_id))
    print('position:', board.pwm_servo_read_position(servo_id))

def main():
    board = Board()
    board.enable_reception()
    # Run each effect once, then leave the strip off.
    rgb_modes = ((4, (255, 0, 0)), (5, (0, 0, 255)))
    rgb_count = 4
    rgb_brightness = 255
    rgb_index = 0
    rgb_finished = False
    next_rgb = time.monotonic()
    print('Ultrasonic IDs 1..4: (id, distance_mm); 65535 = invalid', flush=True)
    # Optional LED example: all four modules, orange breathing, 2-second cycle.
    board.set_ultrasonic_led(0, 1, (64, 24, 0), period_ms=2000)
    last_report = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if not rgb_finished and now >= next_rgb:
                if rgb_index < len(rgb_modes):
                    mode, color = rgb_modes[rgb_index]
                    board.set_rgb(count=rgb_count, mode=mode, rgb=color, brightness=rgb_brightness, period_ms=3000)
                    print('RGB mode %d: %s' % (mode, 'chase' if mode == 4 else 'center expand'), flush=True)
                    rgb_index += 1
                    next_rgb = time.monotonic() + 3.0
                else:
                    board.set_rgb(count=rgb_count, mode=0)
                    rgb_finished = True
                    print('RGB demo finished; strip off', flush=True)
            data = board.get_ultrasonic()
            if data is not None:
                print(data, flush=True)
                last_report = time.monotonic()
            elif time.monotonic() - last_report >= 1.0:
                print('Waiting for ultrasonic report...', flush=True)
                last_report = time.monotonic()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if not rgb_finished:
                board.set_rgb(count=rgb_count, mode=0)
        finally:
            board.enable_reception(False)


if __name__ == "__main__":
    # Standalone use selects explicit wheel commands instead of the legacy RGB demo.
    from .agv_wheels import main as wheels_main
    raise SystemExit(wheels_main())
