"""Bounded diagnostics selected by differences between SQL drafts; never gold.

Supports flat SELECTs and conjunctive equality joins only. Whole-column key
cardinality and declared FKs are evidence, not proof of intended query meaning.
"""
from dataclasses import asdict, dataclass, field
import json
import math
import time
import sqlglot
from sqlglot import exp
from veriquery.sql_execution import SQLiteExecutor


@dataclass
class DisagreementEvidence:
    differences: list[str] = field(default_factory=list)
    clauses: dict = field(default_factory=dict)
    probes: list[dict] = field(default_factory=list)
    schema_reads: int = 0
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def sql_calls(self):
        return len(self.probes)

    def to_dict(self):
        return {**asdict(self), 'sql_calls': self.sql_calls}

    def feedback(self):
        return ('Targeted disagreement evidence, not proof of correctness. Missing FKs do not invalidate joins.\n'
                + json.dumps(self.to_dict(), ensure_ascii=False))


def _describe(sql):
    statements = sqlglot.parse(sql, read='sqlite')
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError('Only one flat SELECT is supported')
    ast = statements[0]
    if ast.args.get('with_') or any(ast.find_all(exp.Subquery)) or len(list(ast.find_all(exp.Select))) != 1:
        raise ValueError('CTEs and nested queries unsupported; no probe')
    sources = [ast.args['from_'].this] if ast.args.get('from_') else []
    sources += [j.this for j in ast.args.get('joins', [])]
    if not sources or any(not isinstance(t, exp.Table) or t.db or t.catalog for t in sources):
        raise ValueError('Only local named tables supported')
    aliases = {t.alias_or_name.lower(): t.name.lower() for t in sources}
    if len(aliases) != len(sources) or len(set(aliases.values())) != len(sources):
        raise ValueError('Duplicate aliases or self joins unsupported; no probe')

    def resolve(c):
        table = aliases.get(c.table.lower()) if c.table else next(iter(aliases.values())) if len(aliases) == 1 else None
        if table is None:
            raise ValueError('Unresolved or ambiguous column; no probe')
        return table, c.name.lower()

    def canonical(node):
        if node is None:
            return ''
        def replace(n):
            if isinstance(n, exp.Column):
                t, c = resolve(n)
                return exp.column(c, table=t, quoted=True)
            return n
        return node.copy().transform(replace).sql(dialect='sqlite', normalize=True)

    edges, joins = [], []
    for join in ast.args.get('joins', []):
        on = join.args.get('on')
        if on is None:
            raise ValueError('USING, NATURAL and cross joins unsupported; no probe')
        terms = list(on.flatten()) if isinstance(on, exp.And) else [on]
        for term in terms:
            if not isinstance(term, exp.EQ) or not isinstance(term.this, exp.Column) or not isinstance(term.expression, exp.Column):
                raise ValueError('Only conjunctive column-equality joins supported; no probe')
            edges.append(tuple(sorted((resolve(term.this), resolve(term.expression)))))
        joins.append((join.args.get('side') or '', join.args.get('kind') or '', canonical(on)))
    return {'joins': sorted(joins),
            'aggregation': [sorted(canonical(a) for a in ast.find_all(exp.AggFunc)), canonical(ast.args.get('group'))],
            'filters': [canonical(ast.args.get('where')), canonical(ast.args.get('having'))], 'edges': set(edges)}


def collect_disagreement_evidence(db_path, first_sql, second_sql, *, max_probes=4, timeout_seconds=1.0):
    """At most one schema read plus max_probes bounded SELECTs.

    Join differences select their changed keys; aggregation differences also
    select shared join keys. Filter differences are explicitly unprobed. No
    query winner or calibrated confidence is inferred from these diagnostics.
    """
    if isinstance(max_probes, bool) or not isinstance(max_probes, int) or not 0 <= max_probes <= 16:
        raise ValueError('max_probes must be an integer between 0 and 16')
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be finite and positive')
    started = time.monotonic()
    out = DisagreementEvidence()
    try:
        a, b = _describe(first_sql), _describe(second_sql)
        for key in ('joins', 'aggregation', 'filters'):
            if a[key] != b[key]:
                out.differences.append(key)
                out.clauses[key] = {'first': a[key], 'second': b[key]}
        edges = a['edges'] ^ b['edges']
        if 'aggregation' in out.differences:
            edges |= a['edges'] | b['edges']
        keys = sorted({key for edge in edges for key in edge})
        if 'filters' in out.differences:
            out.notes.append('Filter disagreement classified; value/selectivity probes not implemented.')
        if not keys or max_probes == 0:
            out.notes.append('No supported key probes selected.' if not keys else 'Probe budget is zero.')
            return out
        executor = SQLiteExecutor(db_path, timeout_seconds=timeout_seconds, max_rows=1)
        out.schema_reads = 1
        cards = {c['table'].lower(): c for c in executor.schema_cards()}
        quote = lambda name: '"' + name.replace('"', '""') + '"'
        for table, col in keys[:max_probes]:
            card = cards.get(table)
            if not card or col not in {c['name'].lower() for c in card['columns']}:
                out.notes.append(f'Unknown key {table}.{col}; no probe.')
                continue
            sql = (f'SELECT COUNT(*) AS rows_total, COUNT({quote(col)}) AS nonnull_rows, '
                   f'COUNT(DISTINCT {quote(col)}) AS distinct_nonnull FROM {quote(table)}')
            result = executor.execute(sql)
            probe = {'table': table, 'column': col, 'sql': sql, 'elapsed_seconds': result.elapsed_seconds,
                     'error': result.error, 'truncated': result.truncated,
                     'declared_foreign_keys': [fk for fk in card['foreign_keys'] if fk['from'].lower() == col]}
            if result.ok and not result.truncated and len(result.rows) == 1:
                total, nonnull, distinct = result.rows[0]
                probe.update(rows_total=total, null_rows=total-nonnull, nonnull_rows=nonnull,
                             distinct_nonnull=distinct, duplicate_nonnull_rows=nonnull-distinct)
            out.probes.append(probe)
        if len(keys) > max_probes:
            out.notes.append(f'Probe budget limited diagnostics to {max_probes} of {len(keys)} keys.')
        out.notes.append('Whole-column cardinality and declared FKs are evidence, not proof of intended join or aggregation.')
    except Exception as exc:
        out.notes.append(f'No further probes: {type(exc).__name__}: {exc}')
    finally:
        out.elapsed_seconds = time.monotonic() - started
    return out
