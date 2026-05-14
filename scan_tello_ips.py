import socket
import time


TELLO_PORT = 8889
IP_PREFIX = "192.168.0."
START_HOST = 101
END_HOST = 130
TIMEOUT_SECONDS = 3
LOCAL_PORT = 9000


def receive_from_ip(sock, expected_ip, timeout_seconds):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        remaining = deadline - time.time()
        sock.settimeout(max(0.1, remaining))
        response, addr = sock.recvfrom(1024)
        if addr[0] != expected_ip:
            continue

        decoded = response.decode("utf-8", errors="ignore").strip()
        return decoded, addr

    raise socket.timeout()


def scan_tello_ips():
    found = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", LOCAL_PORT))

    try:
        print(f"Scanning {IP_PREFIX}{START_HOST} to {IP_PREFIX}{END_HOST} ...")
        print(f"Tello control port is fixed at {TELLO_PORT}")
        print("Validation rule: only count devices that reply to 'command' and return a numeric battery level.")
        print("-" * 50)

        for host in range(START_HOST, END_HOST + 1):
            ip = f"{IP_PREFIX}{host}"
            print(f"Checking {ip}:{TELLO_PORT} ...", end=" ", flush=True)

            try:
                sock.sendto(b"command", (ip, TELLO_PORT))
                command_response, command_addr = receive_from_ip(sock, ip, TIMEOUT_SECONDS)
                if command_response.lower() != "ok":
                    print(
                        f"command replied '{command_response}', not accepted, "
                        f"source={command_addr[0]}:{command_addr[1]}"
                    )
                    time.sleep(0.5)
                    continue

                sock.sendto(b"battery?", (ip, TELLO_PORT))
                battery_response, battery_addr = receive_from_ip(sock, ip, TIMEOUT_SECONDS)

                try:
                    battery_level = int(battery_response)
                except ValueError:
                    print(
                        f"command ok but battery reply invalid: '{battery_response}', "
                        f"source={battery_addr[0]}:{battery_addr[1]}"
                    )
                    time.sleep(0.5)
                    continue

                print(
                    f"FOUND, battery={battery_level}%, "
                    f"source={battery_addr[0]}:{battery_addr[1]}"
                )
                found.append((ip, battery_level, battery_addr[1]))
            except socket.timeout:
                print("no valid tello response")
            except Exception as e:
                print(f"error: {e}")

            time.sleep(0.5)
    finally:
        sock.close()

    print("\nScan complete.")
    if found:
        print("Validated Tello devices:")
        for ip, battery_level, source_port in found:
            print(f"  {ip}:{TELLO_PORT} -> battery={battery_level}%, source_port={source_port}")
    else:
        print("No validated Tello device found in this IP range.")


if __name__ == "__main__":
    scan_tello_ips()
