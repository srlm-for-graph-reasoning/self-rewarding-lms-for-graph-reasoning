#!/usr/bin/env python3
# launches the cypherbench dbs and saves them in the same format as start_neo4j.py

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
import csv
import shlex
import re
from pathlib import Path
from typing import Any, Iterable, Generator

# --- CONFIGURATION ---
class HPCConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent

    # CypherBench Source Data
    BENCHMARK_ROOT = BASE_DIR / "cypherbench" /"benchmark" / "env"
    PROFILE = "full"  # 'full' or 'sampled'

    if PROFILE == "sampled":
        GRAPHS_DIR = BENCHMARK_ROOT / "graphs" / "simplekg_sampled"
    else:
        GRAPHS_DIR = BENCHMARK_ROOT / "graphs" / "simplekg"

    # Serverless Registry on shared filesystem (Lustre/NFS)
    REGISTRY_DIR = BASE_DIR / "cypherbench" / "cluster_registry"

    # Linux RAM Disk for zero I/O latency
    RAM_DISK_DIR = Path("/dev/shm") / f"neo4j_run_{uuid.uuid4().hex[:8]}"

    # Container paths
    ORIGINAL_CONTAINER_PATH = BASE_DIR / "cypherbench" / "envs" / "neo4j_v5.sif"
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


# --- CYPHERBENCH TYPE MAPPING & PARSING ---
TYPE_ALIASES = {
    "str": "string", "string": "string", "text": "string",
    "int": "int", "integer": "int",
    "float": "float", "double": "float",
    "bool": "boolean", "boolean": "boolean",
    "date": "date", "datetime": "datetime",
    "list[str]": "string[]", "list[string]": "string[]",
    "list[int]": "int[]", "list[integer]": "int[]",
    "list[float]": "float[]", "list[double]": "float[]",
    "list[bool]": "boolean[]", "list[boolean]": "boolean[]",
    "list[date]": "date[]", "list[datetime]": "datetime[]",
    "string[]": "string[]", "int[]": "int[]", "float[]": "float[]",
    "boolean[]": "boolean[]", "date[]": "date[]", "datetime[]": "datetime[]"
}

FIXED_NODE_COLUMNS = [("name", "string"), ("description", "string"), ("aliases", "string[]"), ("provenance", "string[]")]
FIXED_REL_COLUMNS = [("rid", "string"), ("provenance", "string[]")]

def to_import_type(dtype: str) -> str:
    key = dtype.strip().lower()
    if key not in TYPE_ALIASES:
        raise ValueError(f"Unsupported CypherBench datatype: {dtype}")
    return TYPE_ALIASES[key]

def format_scalar(value: Any, dtype: str) -> str:
    if value is None: return ""
    dtype = dtype.strip().lower()
    if dtype in {"date", "datetime"} and hasattr(value, "isoformat"): return value.isoformat()
    if dtype in {"bool", "boolean"}: return "true" if bool(value) else "false"
    return str(value)

def format_array(value: Any, item_dtype: str) -> str:
    if not value or not isinstance(value, list): return ""
    return ";".join(format_scalar(v, item_dtype) for v in value)

def format_typed_value(value: Any, dtype: str) -> str:
    dtype = dtype.strip().lower()
    return format_array(value, dtype[:-2]) if dtype.endswith("[]") else format_scalar(value, dtype)

def load_simplekg_schema(graph_json: dict) -> tuple[dict, dict]:
    schema = graph_json["schema"]
    entity_props = {ent["label"]: dict(ent.get("properties", {})) for ent in schema["entities"]}
    relation_props = {(rel["label"], rel["subj_label"], rel["obj_label"]): dict(rel.get("properties", {})) for rel in schema["relations"]}
    return entity_props, relation_props

# --- GENERATOR-BASED CSV STREAMING (OOM FIX) ---
def write_csv_stream(path: Path, header: list[str], row_generator: Generator[list[str], None, None]) -> None:
    """Streams data rows into the CSV file to prevent memory exhaustion on massive graphs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in row_generator:
            writer.writerow(row)

def build_import_artifacts(graph_json_path: Path, import_dir: Path) -> dict[str, Path]:
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    entity_schema, relation_schema = load_simplekg_schema(graph)

    node_prop_types, rel_prop_types = {}, {}
    for props in entity_schema.values():
        for prop, dtype in props.items(): node_prop_types[prop] = dtype
    for props in relation_schema.values():
        for prop, dtype in props.items(): rel_prop_types[prop] = dtype

    node_header = ["eid:ID", ":LABEL"] + [f"{name}:{to_import_type(dtype)}" for name, dtype in FIXED_NODE_COLUMNS] + [f"{prop}:{to_import_type(dtype)}" for prop, dtype in sorted(node_prop_types.items()) if prop not in {"name", "description", "aliases", "provenance"}]

    rel_header = [":START_ID", ":END_ID", ":TYPE"] + [f"{name}:{to_import_type(dtype)}" for name, dtype in FIXED_REL_COLUMNS] + [f"{prop}:{to_import_type(dtype)}" for prop, dtype in sorted(rel_prop_types.items()) if prop not in {"rid", "provenance"}]

    def generate_node_rows() -> Generator[list[str], None, None]:
        for ent in graph["entities"]:
            props = ent.get("properties", {})
            row = [
                str(ent["eid"]), str(ent["label"]), str(ent.get("name", "")),
                format_typed_value(ent.get("description"), "string"),
                format_typed_value(ent.get("aliases", []), "string[]"),
                format_typed_value(ent.get("provenance", []), "string[]")
            ]
            for prop, dtype in sorted(node_prop_types.items()):
                if prop not in {"name", "description", "aliases", "provenance"}:
                    row.append(format_typed_value(props.get(prop), dtype))
            yield row

    def generate_rel_rows() -> Generator[list[str], None, None]:
        for rel in graph["relations"]:
            props = rel.get("properties", {})
            row = [
                str(rel["subj_id"]), str(rel["obj_id"]), str(rel["label"]),
                str(rel.get("rid", "")), format_typed_value(rel.get("provenance", []), "string[]")
            ]
            for prop, dtype in sorted(rel_prop_types.items()):
                if prop not in {"rid", "provenance"}:
                    row.append(format_typed_value(props.get(prop), dtype))
            yield row

    node_data_file = import_dir / "nodes.csv"
    rel_data_file = import_dir / "relationships.csv"

    # Stream data (this naturally writes the header on row 1)
    write_csv_stream(node_data_file, node_header, generate_node_rows())
    write_csv_stream(rel_data_file, rel_header, generate_rel_rows())

    return {"node_data": node_data_file, "rel_data": rel_data_file}

# --- HPC CLUSTER LOGIC ---
def write_readonly_neo4j_conf(conf_file: Path, bolt_port: int, http_port: int, host_ip: str, db_size_mb: int):
    pagecache_size = max(50, db_size_mb * 3)
    conf_content = f"""
server.default_listen_address=0.0.0.0
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
server.directories.import=/var/lib/neo4j/import
"""
    with open(conf_file, "w") as f:
        f.write(conf_content)

async def wait_for_db_online(http_port: int, db_name: str, timeout: int = 240) -> bool:
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        try:
            url = f"http://127.0.0.1:{http_port}/db/neo4j/tx/commit"
            req = urllib.request.Request(
                url,
                data=json.dumps({"statements": [{"statement": "RETURN 1"}]}).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200: return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False

async def monitor_resources(interval: int = 30):
    while True:
        try:
            load1, load5, load15 = os.getloadavg()
            total, used, free = shutil.disk_usage(config.RAM_DISK_DIR)
            used_gb, total_gb = used / (1024**3), total / (1024**3)

            mem_total = mem_available = 0
            if Path('/proc/meminfo').exists():
                for line in Path('/proc/meminfo').read_text().splitlines():
                    if line.startswith('MemTotal:'): mem_total = int(line.split()[1]) / 1024 / 1024
                    elif line.startswith('MemAvailable:'): mem_available = int(line.split()[1]) / 1024 / 1024

            ram_used_gb = mem_total - mem_available
            print(f"{Terminal.BLUE}[Resource Monitor]{Terminal.ENDC} Load: {load1:.2f}, {load5:.2f}, {load15:.2f} | "
                  f"Sys RAM: {ram_used_gb:.1f}G/{mem_total:.1f}G | RAM Disk (/dev/shm): {used_gb:.1f}G/{total_gb:.1f}G")
        except Exception as e:
            print(f"{Terminal.WARNING}[Resource Monitor] Failed to collect stats: {e}{Terminal.ENDC}")
        await asyncio.sleep(interval)

async def boot_instance(graph_json_path: Path, bolt_port: int, host_ip: str, estimated_size_bytes: int):
    name = graph_json_path.name.replace('_simplekg.json', '').replace('_sampled', '')
    http_port = bolt_port + 1
    db_size_mb = max(1, estimated_size_bytes // (1024 * 1024))

    base_dir = config.RAM_DISK_DIR / name
    conf_dir, logs_dir, plugins_dir, import_dir, data_dir = base_dir / "conf", base_dir / "logs", base_dir / "plugins", base_dir / "import", base_dir / "data"

    for d in [conf_dir, logs_dir, plugins_dir, import_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    conf_file = conf_dir / "neo4j.conf"
    write_readonly_neo4j_conf(conf_file, bolt_port, http_port, host_ip, db_size_mb)

    print(f"{Terminal.CYAN}[{name}]{Terminal.ENDC} Translating JSON to CSV in RAM disk...")
    artifacts = await asyncio.to_thread(build_import_artifacts, graph_json_path, import_dir)

    print(f"{Terminal.CYAN}[{name}]{Terminal.ENDC} Executing Neo4j offline bulk import...")
    import_cmd = [
        "apptainer", "exec", "--writable-tmpfs",
        "--bind", f"{conf_file}:/var/lib/neo4j/conf/neo4j.conf",
        "--bind", f"{import_dir}:/var/lib/neo4j/import",
        "--bind", f"{data_dir}:/var/lib/neo4j/data",
        str(config.CONTAINER_PATH),
        "neo4j-admin", "database", "import", "full",
        "--overwrite-destination=true", "--id-type=string", "--array-delimiter=;", "--delimiter=,",
        # Pass the single CSV files which contain both headers and data
        f"--nodes=/var/lib/neo4j/import/nodes.csv",
        f"--relationships=/var/lib/neo4j/import/relationships.csv",
        "neo4j"
    ]

    import_proc = await asyncio.create_subprocess_exec(*import_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await import_proc.communicate()
    if import_proc.returncode != 0:
        print(f"{Terminal.FAIL}[{name}] Import Failed: {stderr.decode()}{Terminal.ENDC}")
        return None, None, None

    # Optional: Clean up import CSVs to save RAM disk space
    for f in import_dir.glob("*.csv"):
        f.unlink()

    print(f"{Terminal.CYAN}[{name}]{Terminal.ENDC} Copying APOC plugins...")
    apoc_cp_cmd = ["apptainer", "exec", "--writable-tmpfs", "--bind", f"{plugins_dir}:/var/lib/neo4j/plugins", str(config.CONTAINER_PATH), "/bin/bash", "-c", "cp /var/lib/neo4j/labs/apoc*.jar /var/lib/neo4j/plugins/"]
    await (await asyncio.create_subprocess_exec(*apoc_cp_cmd)).wait()

    print(f"{Terminal.CYAN}[{name}]{Terminal.ENDC} Booting container on {host_ip}:{bolt_port}...")
    start_cmd = [
        "apptainer", "exec", "--writable-tmpfs",
        "--bind", f"{data_dir}:/var/lib/neo4j/data",
        "--bind", f"{logs_dir}:/var/lib/neo4j/logs",
        "--bind", f"{conf_file}:/var/lib/neo4j/conf/neo4j.conf",
        "--bind", f"{plugins_dir}:/var/lib/neo4j/plugins",
        str(config.CONTAINER_PATH), "neo4j", "console"
    ]

    console_log = open(logs_dir / "console.log", "w")
    server_proc = await asyncio.create_subprocess_exec(*start_cmd, stdout=console_log, stderr=console_log)

    is_online = await wait_for_db_online(http_port, name)
    registry_file = config.REGISTRY_DIR / f"{name}_{host_ip}_{bolt_port}.json"

    if is_online:
        with open(registry_file, "w") as f:
            json.dump({
                "db_name": name,
                "uri": f"bolt://{host_ip}:{bolt_port}",
                "http": f"http://{host_ip}:{http_port}",
                "node": host_ip
            }, f)
        print(f"{Terminal.GREEN}[{name}] ONLINE.{Terminal.ENDC} Registered -> {registry_file.name}")
    else:
        print(f"{Terminal.FAIL}[{name}] FAILED TO BOOT.{Terminal.ENDC} Check {logs_dir}/console.log")
        try: server_proc.kill()
        except ProcessLookupError: pass

    return server_proc, registry_file, console_log

async def main():
    config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    config.RAM_DISK_DIR.mkdir(parents=True, exist_ok=True)

    host_ip = socket.gethostbyname(socket.gethostname())
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    total_nodes = int(os.environ.get("SLURM_NNODES", 1))

    print(f"{Terminal.BOLD}--- INITIALIZING CYPHERBENCH FLEET WORKER (Node {node_id+1}/{total_nodes}) ---{Terminal.ENDC}")
    print(f"Host IP: {host_ip} | RAM Disk: {config.RAM_DISK_DIR}")

    if not config.CONTAINER_PATH.exists():
        print("Staging Apptainer container to RAM disk...")
        await asyncio.to_thread(shutil.copyfile, config.ORIGINAL_CONTAINER_PATH, config.CONTAINER_PATH)

    all_json_graphs = list(config.GRAPHS_DIR.glob("*_simplekg.json"))
    if not all_json_graphs:
        print(f"{Terminal.FAIL}No CypherBench JSON graphs found in {config.GRAPHS_DIR}{Terminal.ENDC}")
        sys.exit(1)

    print(f"Calculating estimated database sizes for {len(all_json_graphs)} workloads...")
    json_sizes = {tb: tb.stat().st_size for tb in all_json_graphs}
    graphs_by_size = sorted(all_json_graphs, key=lambda p: json_sizes[p], reverse=True)

    MAX_CONTAINERS_PER_NODE = 12
    BASE_CONTAINER_OVERHEAD_BYTES = 500 * 1024 * 1024
    node_buckets, node_weights = {i: [] for i in range(total_nodes)}, {i: 0 for i in range(total_nodes)}

    for tb in graphs_by_size:
        available_nodes = [n for n in node_weights.keys() if len(node_buckets[n]) < MAX_CONTAINERS_PER_NODE]
        if not available_nodes:
            print(f"{Terminal.FAIL}Cluster capacity exceeded! Reached hard cap of {MAX_CONTAINERS_PER_NODE} DBs per node.{Terminal.ENDC}")
            sys.exit(1)

        lightest_node = min(available_nodes, key=lambda n: node_weights[n])
        node_buckets[lightest_node].append(tb)
        # Assuming Neo4j DB footprint is roughly 2x the raw JSON size plus overhead
        node_weights[lightest_node] += ((json_sizes[tb] * 2) + BASE_CONTAINER_OVERHEAD_BYTES)

    my_graphs = node_buckets[node_id]
    print(f"Assigned {len(my_graphs)} databases (Estimated Resource Footprint: {node_weights[node_id] // (1024 * 1024)} MB) to this node.")

    tasks, base_port = [], 7550
    for i, graph_path in enumerate(my_graphs):
        tasks.append(boot_instance(graph_path, base_port + (i * 2), host_ip, json_sizes[graph_path] * 2))

    results = await asyncio.gather(*tasks)
    monitor_task = asyncio.create_task(monitor_resources(interval=30))
    shutdown_event = asyncio.Event()

    def handle_sigterm():
        print(f"\n{Terminal.WARNING}Received SIGTERM from Slurm. Initiating shutdown...{Terminal.ENDC}")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(sig, handle_sigterm)

    try:
        print(f"\n{Terminal.BOLD}Node {node_id} idle. Databases are serving traffic.{Terminal.ENDC}")
        await shutdown_event.wait()
    finally:
        print("Cleaning up processes and registry...")
        monitor_task.cancel()

        for result in results:
            if not result: continue
            proc, reg_file, log_file = result
            if proc:
                try: proc.kill()
                except Exception: pass
            if reg_file and reg_file.exists(): reg_file.unlink()
            if log_file: log_file.close()

        print("Wiping RAM disk...")
        if config.RAM_DISK_DIR.exists(): shutil.rmtree(config.RAM_DISK_DIR)
        print("Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
