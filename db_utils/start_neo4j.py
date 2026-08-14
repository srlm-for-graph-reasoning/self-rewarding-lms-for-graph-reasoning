"""
Deploys databases from ../db_tarballs – due to neo4j community edition limitations, the databases need to be run through
individual apptainer instances instead of deploying them together within a single instance.

Due to having to run the script from the base python instance, no external packages can be used.
"""

import json
import socket
import asyncio
import shutil
import tarfile
import uuid
import sys
import os
import signal
import time
import urllib.request
import urllib.error
from pathlib import Path

class HPCConfig:
        BASE_DIR = Path(__file__).resolve().parent.parent

        # Pre-built databases
        TARBALLS_DIR = BASE_DIR / "db_tarballs"

        # DB Registry
        REGISTRY_DIR = BASE_DIR / "cluster_registry"
        RAM_DISK_DIR = Path("/dev/shm") / f"neo4j_run_{uuid.uuid4().hex[:8]}"
        ORIGINAL_CONTAINER_PATH = BASE_DIR / "envs" / "neo4j_v5.sif"
        CONTAINER_PATH = RAM_DISK_DIR / "neo4j_v5.sif"

config = HPCConfig()

class Terminal:
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'

def write_readonly_neo4j_conf(conf_file: Path, bolt_port: int, http_port: int, host_ip: str, db_size_mb: int):
        pagecache_size = max(50, db_size_mb * 3)

        conf_content = f"""
server.bolt.listen_address=0.0.0.0:{bolt_port}
server.http.listen_address=0.0.0.0:{http_port}
server.bolt.advertised_address={host_ip}:{bolt_port}
server.bolt.tls_level=DISABLED
dbms.security.auth_enabled=false

# --- APOC CONFIGURATION ---
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# --- HPC READ-ONLY RAM OPTIMIZATIONS ---
server.memory.pagecache.size={pagecache_size}m
server.memory.heap.initial_size=256m
server.memory.heap.max_size=2G
"""
        with open(conf_file, "w") as f:
                f.write(conf_content)

async def wait_for_db_online(http_port: int, db_name: str, timeout: int = 240) -> bool:
        """Polls the HTTP endpoint to ensure the database is fully booted."""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
                try:
                        url = f"http://127.0.0.1:{http_port}/db/neo4j/tx/commit"
                        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
                        payload = {"statements": [{"statement": "RETURN 1"}]}
                        data = json.dumps(payload).encode('utf-8')

                        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
                        with urllib.request.urlopen(req, timeout=1) as response:
                                if response.status == 200:
                                        return True
                except Exception:
                        pass
                await asyncio.sleep(2)
        return False

async def monitor_resources(interval: int = 30):
        """Background task to periodically log system resource usage to Slurm output."""
        while True:
                try:
                        load1, load5, load15 = os.getloadavg()

                        # RAM disk usage
                        total, used, free = shutil.disk_usage(config.RAM_DISK_DIR)
                        used_gb = used / (1024**3)
                        total_gb = total / (1024**3)

                        # Read system RAM from /proc/meminfo
                        mem_total = mem_available = 0
                        if Path('/proc/meminfo').exists():
                                meminfo = Path('/proc/meminfo').read_text().splitlines()
                                for line in meminfo:
                                        if line.startswith('MemTotal:'):
                                                mem_total = int(line.split()[1]) / 1024 / 1024 # Convert KB to GB
                                        elif line.startswith('MemAvailable:'):
                                                mem_available = int(line.split()[1]) / 1024 / 1024

                        ram_used_gb = mem_total - mem_available

                        print(f"{Terminal.BLUE}[Resource Monitor]{Terminal.ENDC} "
                              f"Load Avg: {load1:.2f}, {load5:.2f}, {load15:.2f} | "
                              f"Sys RAM: {ram_used_gb:.1f}G/{mem_total:.1f}G used | "
                              f"RAM Disk (/dev/shm): {used_gb:.1f}G/{total_gb:.1f}G used")
                except Exception as e:
                        print(f"{Terminal.WARNING}[Resource Monitor] Failed to collect stats: {e}{Terminal.ENDC}")

                await asyncio.sleep(interval)

async def boot_instance(tarball: Path, bolt_port: int, host_ip: str, uncompressed_size_bytes: int):
        name = tarball.name.replace('.tar.gz', '')
        http_port = bolt_port + 1

        db_size_mb = max(1, uncompressed_size_bytes // (1024 * 1024))

        base_dir = config.RAM_DISK_DIR / name
        conf_dir = base_dir / "conf"
        logs_dir = base_dir / "logs"
        plugins_dir = base_dir / "plugins"

        for d in [conf_dir, logs_dir, plugins_dir]:
                d.mkdir(parents=True, exist_ok=True)

        print(f"{Terminal.CYAN}[{name}]{Terminal.ENDC} Extracting {tarball.name} ({db_size_mb}MB uncompressed) to RAM disk...")
        def extract_tar():
                with tarfile.open(tarball, "r:gz") as tar:
                        tar.extractall(path=base_dir)

        await asyncio.to_thread(extract_tar)
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        conf_file = conf_dir / "neo4j.conf"
        write_readonly_neo4j_conf(conf_file, bolt_port, http_port, host_ip, db_size_mb)

        apoc_cp_cmd = [
                "apptainer", "exec", "--writable-tmpfs",
                "--bind", f"{plugins_dir}:/var/lib/neo4j/plugins",
                str(config.CONTAINER_PATH),
                "/bin/bash", "-c", "cp /var/lib/neo4j/labs/apoc*.jar /var/lib/neo4j/plugins/"
        ]
        proc_cp = await asyncio.create_subprocess_exec(*apoc_cp_cmd)
        await proc_cp.wait()

        start_cmd = [
                "apptainer", "exec", "--writable-tmpfs",
                "--bind", f"{data_dir}:/var/lib/neo4j/data",
                "--bind", f"{logs_dir}:/var/lib/neo4j/logs",
                "--bind", f"{conf_file}:/var/lib/neo4j/conf/neo4j.conf",
                "--bind", f"{plugins_dir}:/var/lib/neo4j/plugins",
                str(config.CONTAINER_PATH), "neo4j", "console"
        ]

        print(f"{Terminal.CYAN}[{name}]{Terminal.ENDC} Booting container on {host_ip}:{bolt_port}...")

        console_log = open(logs_dir / "console.log", "w")

        server_proc = await asyncio.create_subprocess_exec(
                *start_cmd, stdout=console_log, stderr=console_log
        )

        is_online = await wait_for_db_online(http_port, name)
        registry_file = config.REGISTRY_DIR / f"{name}_{host_ip}_{bolt_port}.json"

        if is_online:
                registry_data = {
                        "db_name": name,
                        "uri": f"bolt://{host_ip}:{bolt_port}",
                        "http": f"http://{host_ip}:{http_port}",
                        "node": host_ip
                }

                with open(registry_file, "w") as f:
                        json.dump(registry_data, f)

                print(f"{Terminal.GREEN}[{name}] ONLINE.{Terminal.ENDC} Registered -> {registry_file.name}")
        else:
                print(f"{Terminal.FAIL}[{name}] FAILED TO BOOT.{Terminal.ENDC} Check {logs_dir}/console.log")
                try:
                        server_proc.kill()
                except ProcessLookupError:
                        pass

        return server_proc, registry_file, console_log

def get_uncompressed_tar_size(tarball_path: Path) -> int:
        """Parses tar headers to sum the uncompressed sizes of all members without extracting."""
        total_size = 0
        with tarfile.open(tarball_path, "r:gz") as tar:
                for member in tar.getmembers():
                        total_size += member.size
        return total_size

async def main():
        config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        config.RAM_DISK_DIR.mkdir(parents=True, exist_ok=True)

        host_ip = socket.gethostbyname(socket.gethostname())

        node_id = int(os.environ.get("SLURM_NODEID", 0))
        total_nodes = int(os.environ.get("SLURM_NNODES", 1))

        print(f"{Terminal.BOLD}--- INITIALIZING DATABASE FLEET WORKER (Node {node_id+1}/{total_nodes}) ---{Terminal.ENDC}")
        print(f"Host IP: {host_ip}")
        print(f"RAM Disk: {config.RAM_DISK_DIR}")

        if not config.CONTAINER_PATH.exists():
                print("Staging Apptainer container to RAM disk...")
                await asyncio.to_thread(shutil.copyfile, config.ORIGINAL_CONTAINER_PATH, config.CONTAINER_PATH)

        all_tarballs = list(config.TARBALLS_DIR.glob("*.tar.gz"))
        if not all_tarballs:
                print(f"{Terminal.FAIL}No .tar.gz databases found in {config.TARBALLS_DIR}{Terminal.ENDC}")
                sys.exit(1)

        print(f"Calculating true uncompressed database sizes for {len(all_tarballs)} workloads...")
        tarball_sizes = {}
        for tb in all_tarballs:
                tarball_sizes[tb] = get_uncompressed_tar_size(tb)

        tarballs_by_size = sorted(all_tarballs, key=lambda p: tarball_sizes[p], reverse=True)

        MAX_CONTAINERS_PER_NODE = 8
        BASE_CONTAINER_OVERHEAD_BYTES = 500 * 1024 * 1024 # 500 MB base footprint per JVM/Apptainer instance

        node_buckets = {i: [] for i in range(total_nodes)}
        node_weights = {i: 0 for i in range(total_nodes)}

        for tb in tarballs_by_size:
                available_nodes = [n for n in node_weights.keys() if len(node_buckets[n]) < MAX_CONTAINERS_PER_NODE]

                if not available_nodes:
                        print(f"{Terminal.FAIL}Cluster capacity exceeded! Reached hard cap of {MAX_CONTAINERS_PER_NODE} DBs per node.{Terminal.ENDC}")
                        sys.exit(1)

                # Assign to the node with the lightest total overhead/storage footprint
                lightest_node = min(available_nodes, key=lambda n: node_weights[n])

                node_buckets[lightest_node].append(tb)
                node_weights[lightest_node] += (tarball_sizes[tb] + BASE_CONTAINER_OVERHEAD_BYTES)

        my_tarballs = node_buckets[node_id]

        total_mb_assigned = node_weights[node_id] // (1024 * 1024)
        print(f"Assigned {len(my_tarballs)} databases (Estimated Resource Footprint: {total_mb_assigned} MB) to this node.")

        tasks = []
        base_port = 7550
        for i, tarball in enumerate(my_tarballs):
                bolt_port = base_port + (i * 2)
                tasks.append(boot_instance(tarball, bolt_port, host_ip, tarball_sizes[tarball]))

        results = await asyncio.gather(*tasks)


        monitor_task = asyncio.create_task(monitor_resources(interval=30))
        shutdown_event = asyncio.Event()

        def handle_sigterm():
                print(f"\n{Terminal.WARNING}Received SIGTERM from Slurm. Initiating shutdown...{Terminal.ENDC}")
                shutdown_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_sigterm)

        try:
                print(f"\n{Terminal.BOLD}Node {node_id} idle. Databases are serving traffic.{Terminal.ENDC}")
                await shutdown_event.wait()
        finally:
                print("Cleaning up processes and registry...")
                monitor_task.cancel()

                for proc, reg_file, log_file in results:
                        try:
                                proc.kill()
                        except Exception:
                                pass

                        if reg_file.exists():
                                reg_file.unlink()

                        log_file.close()

                print("Wiping RAM disk...")
                if config.RAM_DISK_DIR.exists():
                        shutil.rmtree(config.RAM_DISK_DIR)

                print("Shutdown complete.")

if __name__ == "__main__":
        asyncio.run(main())
