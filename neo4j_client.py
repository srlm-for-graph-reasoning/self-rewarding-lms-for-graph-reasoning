import io
import json
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

import neo4j.graph
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from langchain_neo4j import Neo4jGraph


def clean_cypher(cypher):
    return cypher.replace("\\n", "\n").replace("\\t", " ")


class AsyncNeo4jFleetClient:
    def __init__(self, registry_dir: str | Path, schemas_dir: str | Path = "schemas"):
        self.registry_dir = Path(registry_dir)
        self.schemas_dir = Path(schemas_dir)
        self._drivers = {}
        self._db_map = {}
        self._schema_cache = {}  # Cache for unshuffled schemas
        
        # Ensure the schemas directory exists for deterministic disk caching
        self.schemas_dir.mkdir(parents=True, exist_ok=True)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(self._init_async())
        self._refresh_registry_sync()

    async def _init_async(self):
        self._lock = asyncio.Lock()

    def _refresh_registry_sync(self):
        if not self.registry_dir.exists():
            raise FileNotFoundError(f"Registry directory missing: {self.registry_dir}")

        for reg_file in self.registry_dir.glob("*.json"):
            with open(reg_file, 'r') as f:
                data = json.load(f)
                db_name = data["db_name"]
                uri = data["uri"]
                self._db_map[db_name] = uri

        # Cache schemas for any newly discovered databases
        for db_name, uri in self._db_map.items():
            if db_name not in self._schema_cache:
                if (schema_file:= self.schemas_dir / f"{db_name}_schema.txt").exists():
                    try:
                        with open(schema_file, 'r', encoding='utf-8') as f:
                            self._schema_cache[db_name] = f.read()
                    except Exception as e:
                        print(f"[Warning] Failed to load cached schema from disk for '{db_name}': {e}")
                else:
                    try:
                        graph = Neo4jGraph(
                            url=uri,
                            username="placeholder", # dbs dont have auth but langchain requires these fields to not be empty
                            password="placeholder",
                            enhanced_schema=True
                        )
                        extracted_schema = graph.get_schema
                        self._schema_cache[db_name] = extracted_schema
                        
                        # Save the schema to disk for future deterministic retrieval
                        with open(schema_file, 'w', encoding='utf-8') as f:
                            f.write(extracted_schema)
                    except Exception as e:
                        print(f"[Warning] Failed to query and cache schema for '{db_name}' at {uri}: {e}")

    def get_schema(self, db_name: str, shuffle: bool = True) -> str:
        if db_name not in self._schema_cache:
            self._refresh_registry_sync()
            if db_name not in self._schema_cache:
                raise ValueError(f"Database '{db_name}' not found or schema could not be extracted.")

        schema_str = self._schema_cache[db_name]

        if shuffle:
            return self._shuffle_schema(schema_str)
        return schema_str

    @staticmethod
    def _shuffle_schema(schema_str: str) -> str:
        lines = schema_str.strip().split('\n')

        parsed_data = {
            "Node properties:": [],
            "Relationship properties:": [],
            "The relationships:": []
        }

        current_section = None
        current_entity = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if line in parsed_data:
                current_section = line
                continue

            if current_section in ["Node properties:", "Relationship properties:"]:
                if line.startswith("- **"):
                    current_entity = {"entity_line": line, "props": []}
                    parsed_data[current_section].append(current_entity)
                else:
                    if current_entity:
                        current_entity["props"].append(line)

            elif current_section == "The relationships:":
                parsed_data[current_section].append(line)

        # Shuffle logic
        for section in ["Node properties:", "Relationship properties:"]:
            for item in parsed_data[section]:
                random.shuffle(item["props"])
            random.shuffle(parsed_data[section])

        random.shuffle(parsed_data["The relationships:"])

        # Reconstruction
        output = []
        if parsed_data["Node properties:"]:
            output.append("Node properties:")
            for item in parsed_data["Node properties:"]:
                output.append(item["entity_line"])
                output.extend(item["props"])

        if parsed_data["Relationship properties:"]:
            output.append("Relationship properties:")
            for item in parsed_data["Relationship properties:"]:
                output.append(item["entity_line"])
                output.extend(item["props"])

        if parsed_data["The relationships:"]:
            output.append("The relationships:")
            output.extend(parsed_data["The relationships:"])

        return "\n".join(output)

    async def _get_driver(self, db_name: str):
        async with self._lock:
            if db_name not in self._db_map:
                self._refresh_registry_sync()
                if db_name not in self._db_map:
                    raise ValueError(f"Database '{db_name}' not found in registry.")

            if db_name not in self._drivers:
                uri = self._db_map[db_name]
                self._drivers[db_name] = AsyncGraphDatabase.driver(
                    uri, auth=None, notifications_min_severity="OFF"
                )

            return self._drivers[db_name]

    def _format_cypher_like(self, val: Any) -> str:
        """
        Recursively converts graph objects and primitives into
        human-readable, syntactically valid Cypher strings.
        """
        if val is None:
            return "null"
        elif isinstance(val, bool):
            return "true" if val else "false"
        elif isinstance(val, (int, float)):
            return str(val)
        elif isinstance(val, str):
            return json.dumps(val, ensure_ascii=False)

        elif isinstance(val, neo4j.graph.Node):
            labels = ":".join(val.labels)
            labels_str = f":{labels}" if labels else ""
            props = ", ".join(f"{k}: {self._format_cypher_like(v)}" for k, v in val.items())
            props_str = f" {{{props}}}" if props else ""
            return f"({labels_str}{props_str})"

        elif isinstance(val, neo4j.graph.Relationship):
            props = ", ".join(f"{k}: {self._format_cypher_like(v)}" for k, v in val.items())
            props_str = f" {{{props}}}" if props else ""
            return f"[:{val.type}{props_str}]"

        elif isinstance(val, neo4j.graph.Path):
            path_str = self._format_cypher_like(val.start_node)
            current_node = val.start_node

            for rel in val.relationships:
                props = ", ".join(f"{k}: {self._format_cypher_like(v)}" for k, v in rel.items())
                props_str = f" {{{props}}}" if props else ""

                if rel.start_node == current_node:
                    next_node = rel.end_node
                    path_str += f"-[:{rel.type}{props_str}]->{self._format_cypher_like(next_node)}"
                else:
                    next_node = rel.start_node
                    path_str += f"<-[:{rel.type}{props_str}]-{self._format_cypher_like(next_node)}"

                current_node = next_node
            return path_str

        elif isinstance(val, list):
            return "[" + ", ".join(self._format_cypher_like(v) for v in val) + "]"

        elif isinstance(val, dict):
            return "{" + ", ".join(f"{k}: {self._format_cypher_like(v)}" for k, v in val.items()) + "}"

        return str(val)

    async def query(self, db_name: str, cypher: str, parameters: Optional[Dict] = None, max_rows: int = 250, max_chars: int = 6_000, fmt: str = "cypher_like") -> str:
        cleaned_cypher = clean_cypher(cypher)

        max_retries = 2
        for attempt in range(max_retries):
            driver = await self._get_driver(db_name)
            try:
                async with driver.session() as session:
                    async def _execute_and_format():
                        output = io.StringIO()
                        result = await session.run(cleaned_cypher, parameters or {})

                        keys = result.keys()
                        row_count = 0
                        truncated = False

                        async for record in result:
                            if row_count >= max_rows or output.tell() >= max_chars:
                                truncated = True
                                break
                            row_count += 1

                            if fmt == "cypher_like":
                                row_strs = [f"{k}: {self._format_cypher_like(record[k])}" for k in keys]
                                output.write(" | ".join(row_strs) + "\n")

                        final_output = output.getvalue()
                        if len(final_output) > max_chars:
                            final_output = final_output[:max_chars] + f"\n... [TRUNCATED] Reached hard char limit ({max_chars})."
                        elif truncated:
                            final_output += f"\n... [TRUNCATED] Reached row limit ({row_count}/{max_rows})."

                        return final_output

                    return await asyncio.wait_for(
                        _execute_and_format(),
                        timeout=90.0
                    )

            except asyncio.TimeoutError:
                raise TimeoutError("Client-side timeout: Query execution exceeded 90 seconds.")
            except (ServiceUnavailable, SessionExpired) as e:
                if attempt < max_retries - 1:
                    print(f"[Client] Database {db_name} unavailable. Reconnecting...")
                    async with self._lock:
                        if db_name in self._drivers:
                            await self._drivers[db_name].close()
                            del self._drivers[db_name]
                    await asyncio.sleep(5)
                    self._refresh_registry_sync()
                else:
                    raise e

    async def close(self):
        async with self._lock:
            for driver in self._drivers.values():
                await driver.close()
            self._drivers.clear()
