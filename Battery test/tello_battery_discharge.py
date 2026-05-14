"""
Tello EDU Single-Drone Battery Discharge Curve Logger
=====================================================

用于采集 LiPo 电池非线性放电曲线 (non-linear discharge curve)，
特别用于定位 70%-80% Hidden Voltage Plateau。

实验流程：
    1. 插入电池，连接 Tello WiFi
    2. 运行脚本（IDE Run 或 python tello_battery_discharge.py）
    3. 提示输入 battery_id (1..99)
    4. 程序自动进入 SDK 模式 → takeoff → 悬停 → 持续记录 state telemetry
    5. 当 battery% ≤ STOP_BATTERY 时自动 land
    6. 数据写入 discharge_logs/battery_{id:02d}_{timestamp}.csv

10 块电池流程：
    每换一块电池，重新 Run 一次脚本，输入对应编号即可。

State telemetry 字段（port 8890, ~10 Hz）：
    pitch, roll, yaw, vgx, vgy, vgz, templ, temph,
    tof, h, bat, baro, time, agx, agy, agz
"""

import csv
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime


# ====================== 实验参数 ======================
TELLO_IP            = '192.168.0.107'
TELLO_CMD_PORT      = 8889
LOCAL_CMD_PORT      = 9000      # 接收命令响应
LOCAL_STATE_PORT    = 8890      # 接收状态遥测

LOG_HZ              = 10        # CSV 记录频率 (Hz)
KEEPALIVE_INTERVAL  = 1.0       # rc 0 0 0 0 间隔 (s) —— 防止 SDK 超时/误触发降落
FLUSH_INTERVAL      = 5.0       # CSV flush + fsync 间隔 (s)
PROGRESS_INTERVAL   = 30.0      # 控制台进度打印间隔 (s)

STOP_BATTERY        = 10        # 终止阈值 (%)，到此触发 land
TAKEOFF_STABILIZE   = 5.0       # 起飞后稳定等待 (s)
LAND_STABILIZE      = 8.0       # 降落后等待 (s)
LANDED_HEIGHT_CM    = 10        # h/tof 连续低于该高度，判定已非预期落地
LANDED_CONFIRM_S    = 2.0

OUTPUT_DIR          = 'discharge_logs'

# CSV 字段定义
CSV_FIELDS = [
    'timestamp_iso', 'elapsed_s', 'battery_pct',
    'templ', 'temph',
    'baro', 'h', 'tof',
    'pitch', 'roll', 'yaw',
    'vgx', 'vgy', 'vgz',
    'agx', 'agy', 'agz',
]


# ====================== 主类 ======================
class TelloDischargeTester:
    def __init__(self, battery_id: int):
        self.battery_id = battery_id

        # Sockets
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(('', LOCAL_CMD_PORT))
        self.cmd_sock.settimeout(1.0)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(('', LOCAL_STATE_PORT))
        self.state_sock.settimeout(1.0)

        # 共享状态
        self.current_state: dict = {}
        self.state_lock = threading.Lock()
        self.last_response: str | None = None

        # 控制标志
        self.running = True
        self.airborne = False

        # 后台线程
        self.state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self.resp_thread  = threading.Thread(target=self._response_loop, daemon=True)

    # ----------- 后台接收线程 -----------
    def _state_loop(self):
        """持续接收 port 8890 的状态遥测"""
        while self.running:
            try:
                data, _ = self.state_sock.recvfrom(1024)
                state = self._parse_state(data.decode('utf-8').strip())
                with self.state_lock:
                    self.current_state = state
            except socket.timeout:
                continue
            except OSError:
                break  # socket 被关闭
            except Exception as e:
                if self.running:
                    print(f'[STATE] parse error: {e}')

    def _response_loop(self):
        """持续接收命令响应"""
        while self.running:
            try:
                data, _ = self.cmd_sock.recvfrom(1024)
                self.last_response = data.decode('utf-8').strip()
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                if self.running:
                    print(f'[CMD] recv error: {e}')

    @staticmethod
    def _parse_state(s: str) -> dict:
        """解析 'pitch:0;roll:0;...' 格式"""
        result = {}
        for kv in s.split(';'):
            if ':' not in kv:
                continue
            k, v = kv.split(':', 1)
            try:
                result[k] = float(v) if '.' in v else int(v)
            except ValueError:
                result[k] = v
        return result

    # ----------- 命令发送 -----------
    def send(self, cmd: str, expect_response: bool = True, timeout: float = 7.0):
        print(f'>> {cmd}')
        self.last_response = None
        self.cmd_sock.sendto(cmd.encode(), (TELLO_IP, TELLO_CMD_PORT))
        if not expect_response:
            return None

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.last_response is not None:
                resp = self.last_response
                print(f'<< {resp}')
                return resp
            time.sleep(0.05)
        print('<< [timeout]')
        return None

    def _send_hover_keepalive(self):
        self.send('rc 0 0 0 0', expect_response=False)

    def _wait_with_keepalive(self, seconds: float):
        deadline = time.time() + seconds
        next_keepalive = time.time()
        while self.running and time.time() < deadline:
            now = time.time()
            if now >= next_keepalive:
                self._send_hover_keepalive()
                next_keepalive = now + KEEPALIVE_INTERVAL
            time.sleep(0.05)

    # ----------- 主流程 -----------
    def run(self):
        self.state_thread.start()
        self.resp_thread.start()

        # 1. 进入 SDK 模式
        if self.send('command') != 'ok':
            raise RuntimeError('无法进入 SDK 模式 — 请确认已连接 Tello WiFi')

        # 2. 等待 state telemetry
        print('[INFO] 等待 state telemetry...')
        for _ in range(50):
            time.sleep(0.1)
            with self.state_lock:
                if self.current_state:
                    break
        else:
            raise RuntimeError('未收到 state telemetry — 请检查 port 8890')

        with self.state_lock:
            init_bat = self.current_state.get('bat', 0)
            init_templ = self.current_state.get('templ', '?')

        print(f'[INFO] Battery #{self.battery_id} | 起始电量 = {init_bat}% | '
              f'起始 templ = {init_templ}°C')

        # 3. 准备 CSV
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(
            OUTPUT_DIR, f'battery_{self.battery_id:02d}_{ts}.csv'
        )

        # 4. Takeoff
        print(f'[INFO] takeoff (battery_id={self.battery_id})')
        if self.send('takeoff', timeout=15) != 'ok':
            raise RuntimeError('takeoff 失败')
        self.airborne = True
        self._send_hover_keepalive()
        self._wait_with_keepalive(TAKEOFF_STABILIZE)

        # 5. Hover + log（异常或终止时一定会执行 land）
        try:
            self._hover_and_log(csv_path, init_bat)
        finally:
            self._safe_land()

    def _hover_and_log(self, csv_path: str, init_bat: int):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)

            t_start = time.time()
            t_last_keepalive = t_start
            t_last_flush     = t_start
            t_last_print     = t_start
            t_landed_since   = None

            sample_interval = 1.0 / LOG_HZ
            t_next_sample = t_start

            print(f'[INFO] 开始记录 → 目标 battery ≤ {STOP_BATTERY}% 自动 land')
            print(f'[INFO] CSV → {csv_path}')

            while self.running:
                now = time.time()

                # 按 LOG_HZ 节流采样
                if now < t_next_sample:
                    time.sleep(min(0.01, t_next_sample - now))
                    continue
                t_next_sample = now + sample_interval

                # 取 snapshot
                with self.state_lock:
                    s = dict(self.current_state)
                if not s:
                    continue

                bat = s.get('bat', init_bat)
                h = s.get('h')
                tof = s.get('tof')
                elapsed = now - t_start

                writer.writerow([
                    datetime.now().isoformat(timespec='milliseconds'),
                    round(elapsed, 3),
                    bat,
                    s.get('templ', ''), s.get('temph', ''),
                    s.get('baro', ''), s.get('h', ''), s.get('tof', ''),
                    s.get('pitch', ''), s.get('roll', ''), s.get('yaw', ''),
                    s.get('vgx', ''), s.get('vgy', ''), s.get('vgz', ''),
                    s.get('agx', ''), s.get('agy', ''), s.get('agz', ''),
                ])

                # rc keep-alive：维持 hover + 防超时
                if now - t_last_keepalive >= KEEPALIVE_INTERVAL:
                    self._send_hover_keepalive()
                    t_last_keepalive = now

                # 周期性 flush
                if now - t_last_flush >= FLUSH_INTERVAL:
                    f.flush()
                    os.fsync(f.fileno())
                    t_last_flush = now

                # 进度打印
                if now - t_last_print >= PROGRESS_INTERVAL:
                    print(
                        f'[t={elapsed:7.1f}s] battery={bat:3d}% '
                        f'h={s.get("h", 0):>3}cm '
                        f'templ={s.get("templ", 0):>2}°C '
                        f'temph={s.get("temph", 0):>2}°C'
                    )
                    t_last_print = now

                # 终止条件
                if bat <= STOP_BATTERY:
                    print(f'[INFO] 电量达到阈值 {bat}% ≤ {STOP_BATTERY}% — land')
                    break

                if elapsed > 2.0 and self._looks_landed(h, tof):
                    if t_landed_since is None:
                        t_landed_since = now
                    elif now - t_landed_since >= LANDED_CONFIRM_S:
                        print(
                            '[WARN] 检测到飞机已贴地，但电量未到阈值；'
                            '这更像是 Tello 自保护/SDK 保活/环境导致的非预期降落'
                        )
                        break
                else:
                    t_landed_since = None

            f.flush()
            os.fsync(f.fileno())

        print(f'[INFO] 数据保存完成 → {csv_path}')

    @staticmethod
    def _looks_landed(h, tof) -> bool:
        try:
            h_low = h is not None and float(h) <= LANDED_HEIGHT_CM
            tof_low = tof is not None and float(tof) <= LANDED_HEIGHT_CM
        except (TypeError, ValueError):
            return False
        return h_low and tof_low

    # ----------- 安全降落 -----------
    def _safe_land(self):
        if not self.airborne:
            return
        print('[INFO] 发送 land 指令')
        self.send('land', timeout=15)
        time.sleep(LAND_STABILIZE)
        self.airborne = False

    def shutdown(self):
        self.running = False
        try:
            self.cmd_sock.close()
            self.state_sock.close()
        except Exception:
            pass


# ====================== 入口 ======================
def main():
    # 交互式输入电池编号（IDE 直接 Run 也能用）
    while True:
        raw = input('请输入电池编号 battery_id (1..99): ').strip()
        try:
            battery_id = int(raw)
            if 1 <= battery_id <= 99:
                break
            print('  ✗ 应在 1..99 之间，请重新输入')
        except ValueError:
            print('  ✗ 必须是整数，请重新输入')

    tester = TelloDischargeTester(battery_id)

    # Ctrl+C 紧急降落
    def sigint_handler(sig, frame):
        print('\n[!] 收到 SIGINT — 紧急 land')
        tester.running = False
        tester._safe_land()
        tester.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        tester.run()
    except Exception as e:
        print(f'[ERROR] {e}')
        tester._safe_land()
        sys.exit(2)
    finally:
        tester.shutdown()


if __name__ == '__main__':
    main()
