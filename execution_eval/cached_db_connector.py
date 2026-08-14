# Adapted Neo4j client with CypherBench-specific integrations and golden query result caching for inter-iteration validation

import json
import time
import asyncio
import re
import copy
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, List, Literal, Any

import pandas as pd
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase

# ==========================================
# Async Neo4j Fleet Connector
# ==========================================
class AsyncNeo4jFleetConnector:
    """
    Async implementation combining connection routing via registry JSONs
    and schema extraction methods adapted for asyncio.
    """
    def __init__(self, registry_dir: str | Path, username=None, password=None,
                 max_connection_pool_size: int = 100, debug=False):
        self.registry_dir = Path(registry_dir)
        self.pool_size = max_connection_pool_size
        self.debug = debug
        self.auth = (username, password) if username and password else None

        self._drivers = {}
        self._db_map = {}
        self._lock = None  # Instantiated lazily in the event loop
        
        # New Cache layer specifically helpful for golden targets
        self._query_cache: Dict[Tuple[str, str, str], Any] = {}

    async def _init_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()

    def _refresh_registry_sync(self):
        if not self.registry_dir.exists():
            print(f"Warning: Registry directory missing: {self.registry_dir}")
            return
        for reg_file in self.registry_dir.glob("*.json"):
            with open(reg_file, 'r') as f:
                data = json.load(f)
                self._db_map[data["db_name"]] = data["uri"]

    async def _get_driver(self, db_name: str):
        await self._init_lock()
        async with self._lock:
            if db_name not in self._db_map:
                self._refresh_registry_sync()
                if db_name not in self._db_map:
                    raise ValueError(f"Database '{db_name}' not found in registry.")

            if db_name not in self._drivers:
                uri = self._db_map[db_name]
                driver = AsyncGraphDatabase.driver(
                    uri,
                    auth=self.auth,
                    max_connection_pool_size=self.pool_size,
                    notifications_min_severity="OFF"
                )
                await driver.verify_connectivity()
                self._drivers[db_name] = driver

            return self._drivers[db_name]

    def clear_cache(self):
        """Clears the connector-level query cache."""
        self._query_cache.clear()

    async def run_query(self, db_name: str, cypher: str, parameters=None, timeout: float = 30.0, use_cache: bool = False):
        # 1. Cache Checking (Read)
        cache_key = (db_name, cypher, str(parameters or {}))
        if use_cache:
            await self._init_lock()
            async with self._lock:
                if cache_key in self._query_cache:
                    # Return deepcopy so downstream logic (like hashing dicts) doesn't pollute the cached reference
                    return copy.deepcopy(self._query_cache[cache_key])

        if self.debug:
            t0 = time.time()
            print(f'[{db_name}] Running Cypher:\n```\n{cypher}\n```')

        driver = await self._get_driver(db_name)

        # We wrap the entire execution in a helper function so asyncio.wait_for
        # can safely manage the session closure if a client-side timeout occurs.
        async def _execute():
            async with driver.session(database="neo4j") as session:
                # session.run natively accepts timeout for auto-commit transactions.
                # This explicitly sets the 30s kill limit on the database server.
                result = await session.run(cypher, parameters or {}, timeout=timeout)
                return await result.data()

        try:
            # Client-Side constraint: 32 seconds (timeout + 2s buffer)
            data = await asyncio.wait_for(_execute(), timeout=timeout + 2.0)

        except asyncio.TimeoutError:
            # Triggered if the network drops and the DB can't communicate the timeout back
            raise TimeoutError(f"Client-side timeout: Server did not respond within {timeout + 2.0}s")
        except Exception as e:
            # Triggered if the DB successfully kills the query at 30 seconds and notifies us
            err_str = str(e)
            if "TransactionTimedOut" in err_str or "DeadlockDetected" in err_str:
                raise TimeoutError(f"Server-side timeout: Neo4j terminated the query after {timeout}s.")
            # Reraise standard Cypher syntax errors or missing databases
            raise e

        # 2. Cache Checking (Write)
        if use_cache:
            async with self._lock:
                self._query_cache[cache_key] = copy.deepcopy(data)

        if self.debug:
            print(f'[{db_name}] Cypher finished in {time.time() - t0:.2f}s')

        return data

    async def close(self):
        await self._init_lock()
        async with self._lock:
            for driver in self._drivers.values():
                await driver.close()
            self._drivers.clear()
            self._query_cache.clear()
