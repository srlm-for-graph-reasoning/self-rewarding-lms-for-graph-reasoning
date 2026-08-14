# Adapted from:
# https://github.com/megagonlabs/cypherbench/blob/main/cypherbench/metrics/provenance_subgraph_jaccard_similarity.py
# https://github.com/megagonlabs/cypherbench/blob/main/cypherbench/metrics/execution_accuracy.py
# https://github.com/megagonlabs/cypherbench/blob/main/cypherbench/metrics/executable.py

import re
import time
import asyncio
import random
import logging
from typing import List, Tuple, Dict, Set, Optional, Any
from itertools import product
from collections import defaultdict

import neo4j
import neo4j.exceptions

logger = logging.getLogger(__name__)

# ==========================================
# Metric 1: Executable
# ==========================================

async def executable(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: "AsyncNeo4jFleetConnector",
    db_name: str,
    timeout: float = 120.0
) -> float:
    """Whether the predicted Cypher query is executable"""
    try:
        await neo4j_connector.run_query(db_name, pred_cypher, timeout=timeout)
    except Exception:
        return 0.0

    return 1.0


# ==========================================
# Metric 2: Execution Accuracy Helpers
# ==========================================

def to_hashable(obj, unorder_list=True):
    """
    Recursively transforms a list, dictionary, or set into a hashable object.
    Lists and sets are converted to tuples. Dictionaries are converted to tuples of sorted (key, value) pairs.
    """
    if isinstance(obj, (tuple, int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (neo4j.time.Date, neo4j.time.DateTime, neo4j.time.Time, neo4j.time.Duration)):
        return obj.iso_format() if hasattr(obj, 'iso_format') else str(obj)
    elif isinstance(obj, (list, tuple)):
        if unorder_list:
            try:
                return tuple(sorted(to_hashable(item) for item in obj))
            except TypeError:
                # Fallback if list contains mixed types (e.g. str and int) that Python can't sort
                return tuple(sorted((to_hashable(item) for item in obj), key=str))
        else:
            return tuple(to_hashable(item) for item in obj)
    elif isinstance(obj, set):
        try:
            return tuple(sorted(to_hashable(item) for item in obj))
        except TypeError:
            return tuple(sorted((to_hashable(item) for item in obj), key=str))
    elif isinstance(obj, dict):
        try:
            return tuple(sorted((to_hashable(k), to_hashable(v)) for k, v in obj.items()))
        except TypeError:
            return tuple(sorted(((to_hashable(k), to_hashable(v)) for k, v in obj.items()), key=lambda x: str(x[0])))
    else:
        return str(obj)

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    for _ in range(20):
        random_tab2_row = random.choice(result2)
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        return False

    if not quick_rej(result1, result2, order_matters):
        return False

    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def to_tuples(result: List[Dict]) -> List[Tuple]:
    keys = list(result[0].keys())
    for row in result:
        assert set(row.keys()) == set(keys)
    return [tuple([row[key] for key in keys]) for row in result]


def _compare_execution(
        pred_executed: list[dict], target_executed: list[dict], order_matters: bool
) -> float:
    if not pred_executed and not target_executed:
        return 1.0
    elif not pred_executed or not target_executed:
        return 0.0

    gold_tuples = to_tuples(target_executed)
    pred_tuples = to_tuples(pred_executed)
    return float(result_eq(gold_tuples, pred_tuples, order_matters=order_matters))


# ==========================================
# Metric 2: Execution Accuracy PRECOMPUTE & EVAL
# ==========================================

async def compute_target_ea(
    target_cypher: str, 
    neo4j_connector: "AsyncNeo4jFleetConnector", 
    db_name: str, 
    timeout: float = 120.0
) -> dict:
    """Computes and hashes the target execution result exactly once."""
    try:
        target_executed = await neo4j_connector.run_query(db_name, target_cypher, timeout=timeout * 2.0, use_cache=True)
        # Hash results immediately so we don't repeat this CPU-heavy task for every candidate
        hashed_executed = [{k: to_hashable(v) for k, v in record.items()} for record in target_executed]
        return {
            "success": True,
            "data": hashed_executed,
            "order_matters": 'order by' in target_cypher.lower()
        }
    except Exception as e:
        return {"success": False, "error": e}


async def execution_accuracy(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: "AsyncNeo4jFleetConnector",
    db_name: str,
    timeout: float = 120.0,
    target_cache: dict = None
) -> float:
    """Execution accuracy for two cypher queries, utilizing target cache if provided."""
    if pred_cypher == target_cypher:
        return 1.0

    # Fallback to computing it if not provided by the orchestrator
    if target_cache is None:
        target_cache = await compute_target_ea(target_cypher, neo4j_connector, db_name, timeout)

    if not target_cache.get("success", False):
        return 0.0 # Target failed, impossible to match

    try:
        pred_executed = await neo4j_connector.run_query(db_name, pred_cypher, timeout=timeout)
        pred_executed = [{k: to_hashable(v) for k, v in record.items()} for record in pred_executed]
    except Exception:
        return 0.0

    return _compare_execution(
        pred_executed=pred_executed,
        target_executed=target_cache["data"],
        order_matters=target_cache["order_matters"]
    )


# ==========================================
# Metric 3: Provenance Subgraph JS Helpers
# ==========================================

def split_by_union(cypher: str) -> List[str]:
    pattern = r'\bUNION\b'
    if cypher.strip().startswith("CALL"):
        inner_query_match = re.search(r'CALL\s*\{(.*?)\}\s*(WITH|RETURN|WHERE|UNWIND)', cypher, re.DOTALL)
        if inner_query_match:
            inner_query = inner_query_match.group(1)
            split_inner_queries = re.split(pattern, inner_query)
            return [q.strip() for q in split_inner_queries]
        else:
            return [cypher.strip()]
    else:
        split_queries = re.split(pattern, cypher)
        return [q.strip() for q in split_queries]


def split_cypher_into_clauses(cypher_query: str) -> list:
    clause_pattern = r'\b(MATCH|OPTIONAL MATCH|WHERE|RETURN|UNION|WITH|CREATE|SET|DELETE|MERGE|UNWIND|ORDER BY|LIMIT|SKIP|FOREACH|CALL|YIELD)\b'
    matches = list(re.finditer(clause_pattern, cypher_query))
    clauses = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cypher_query)
        clauses.append(cypher_query[start:end].strip())
    return clauses


def extract_match_cypher(cypher: str) -> Optional[str]:
    if not cypher.startswith('MATCH'):
        return None
    clauses = split_cypher_into_clauses(cypher)
    match_clauses = []
    for clause in clauses:
        if not any(clause.startswith(keyword) for keyword in ['MATCH', 'OPTIONAL MATCH', 'WITH', 'WHERE']):
            break
        if clause.startswith('WITH'):
            if ' as ' in clause.lower():
                break
            else:
                match_clauses.append('WITH *')
        else:
            match_clauses.append(clause)
    while match_clauses and match_clauses[-1].startswith('WITH'):
        match_clauses.pop()
    return ' '.join(match_clauses)


def add_variables(match_cypher: str) -> str:
    node_counter = 0
    relationship_counter = 0

    def replace_node(match):
        nonlocal node_counter
        replacement = f"(ntmp{node_counter}:{match.group(2)}{match.group(3) or ''})"
        node_counter += 1
        return replacement

    def replace_relationship(match):
        nonlocal relationship_counter
        replacement = f"[rtmp{relationship_counter}{match.group(2)}]"
        relationship_counter += 1
        return replacement

    clauses = split_cypher_into_clauses(match_cypher)
    for i, clause in enumerate(clauses):
        if clause.startswith('MATCH') or clause.startswith('OPTIONAL MATCH'):
            clause = re.sub(r'(\[)(:.*?)(\])', replace_relationship, clause)
            clauses[i] = re.sub(r'(\(:)([A-Za-z]+)(\s*\{.*?\})?\)', replace_node, clause)

    return ' '.join(clauses)


def extract_node_variables(match_cypher: str) -> List[str]:
    match_cypher = re.sub(r'\{[^}]*\}', '{dummy}', match_cypher)
    pattern = r'\((\w+)(?::[^\)]*|\))'
    vars = []
    clauses = split_cypher_into_clauses(match_cypher)
    for clause in clauses:
        if clause.startswith('MATCH') or clause.startswith('OPTIONAL MATCH'):
            vars += re.findall(pattern, clause)
    return sorted(list(set(vars)))


def extract_relationship_variables(match_cypher: str) -> List[str]:
    pattern = r'-\[(\w+)(?::|\])'
    vars = []
    clauses = split_cypher_into_clauses(match_cypher)
    for clause in clauses:
        if clause.startswith('MATCH') or clause.startswith('OPTIONAL MATCH'):
            vars += re.findall(pattern, clause)
    return sorted(list(set(vars)))


def get_ps_cypher(cypher: str, return_var='elemId', node_element_id_only=False) -> str:
    logger.debug(f'Getting provenance subgraph for cypher: {cypher}')
    cyphers = split_by_union(cypher)
    ps_cyphers = []
    for sub_cypher in cyphers:
        match_cypher = extract_match_cypher(sub_cypher)
        if match_cypher:
            match_cypher = add_variables(match_cypher)
            logger.debug(f'match_cypher: {match_cypher}')
            node_vars = extract_node_variables(match_cypher)
            rel_vars = extract_relationship_variables(match_cypher)
            if node_element_id_only:
                node_expr = ' + '.join(f'collect(distinct elementId({var}))' for var in node_vars)
                node_expr = node_expr if node_expr else "[]"
                ps_cyphers.append(
                    f'{match_cypher} WITH {node_expr} AS elemIds UNWIND elemIds AS elemId RETURN elemId AS {return_var}')
            else:
                node_expr = ' + '.join(f'collect(distinct {var})' for var in node_vars)
                node_expr = node_expr if node_expr else "[]"
                rel_expr = ' + '.join(f'collect(distinct {var})' for var in rel_vars)
                rel_expr = rel_expr if rel_expr else "[]"
                ps_cyphers.append(f'{match_cypher} RETURN {node_expr} AS nodes, {rel_expr} AS relationships')

    if len(ps_cyphers) == 0:
        if node_element_id_only:
            ps_cypher = f'UNWIND [] AS elemId RETURN elemId AS {return_var}'
        else:
            ps_cypher = 'RETURN [] AS nodes, [] AS relationships'
    else:
        ps_cypher = ' UNION '.join(ps_cyphers)

    logger.debug(f'Provenance subgraph cypher: {ps_cypher}')
    return ps_cypher


# ==========================================
# Metric 3: Provenance Subgraph JS PRECOMPUTE & EVAL
# ==========================================

async def compute_target_psjs(
    target_cypher: str, 
    neo4j_connector: "AsyncNeo4jFleetConnector", 
    db_name: str, 
    timeout: float = 240.0
) -> dict:
    """Computes the target Subgraph ID set exactly once."""
    target_ps_cypher = get_ps_cypher(target_cypher, node_element_id_only=True, return_var='elemId1')
    try:
        result_target = await neo4j_connector.run_query(db_name, target_ps_cypher, timeout=timeout, use_cache=True)
        target_ps = set(record['elemId1'] for record in result_target)
        return {"success": True, "data": target_ps}
    except Exception as e:
        return {"success": False, "error": e}


async def provenance_subgraph_jaccard_similarity(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: "AsyncNeo4jFleetConnector",
    db_name: str,
    timeout: float = 120.0,
    target_cache: dict = None
) -> float:
    """Provenance Subgraph JS utilizing target cache if provided."""
    if pred_cypher == target_cypher:
        return 1.0

    # Fallback to computing it if not provided by the orchestrator
    if target_cache is None:
        target_cache = await compute_target_psjs(target_cypher, neo4j_connector, db_name, timeout)
        
    if not target_cache.get("success", False):
        return 0.0

    pred_ps_cypher = get_ps_cypher(pred_cypher, node_element_id_only=True, return_var='elemId2')

    try:
        result_pred = await neo4j_connector.run_query(db_name, pred_ps_cypher, timeout=timeout)
        pred_ps = set(record['elemId2'] for record in result_pred)

        target_ps = target_cache["data"]
        I = len(target_ps.intersection(pred_ps))
        U = len(target_ps.union(pred_ps))
        psjs = I / U if U > 0 else 0.0
    except Exception as e:
        return 0.0

    return psjs
