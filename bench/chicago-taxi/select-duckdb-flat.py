import sys
import time

import duckdb

path = sys.argv[1] if len(sys.argv) > 1 else "chicago-taxi-flat.parquet"

# DuckDB reads the Parquet file directly via read_parquet(); the flat schema
# keeps the leaf names dotted ("payment.tips", "trip.begin.lon"), so they must
# be double-quoted to avoid being parsed as nested struct accesses.
# The predicate/projection mirror select-arrow-flat.py so results are comparable.
query = f"""
SELECT
  "payment.tips",
  "payment.total",
  "trip.sec",
  "trip.km",
  company
FROM read_parquet('{path}')
WHERE "payment.tips" > 100 AND "trip.km" > 0 AND "trip.begin.lon" < 0
ORDER BY "trip.sec"
"""

t0 = time.perf_counter()
# In-memory connection; the parquet is opened lazily inside the query so that
# the filter (and row-group/page pruning) happens during "compute", mirroring
# blosc2's where() and arrow's scanner pushdown.
con = duckdb.connect()
t1 = time.perf_counter()

# Materialize the result into a temporary table.  This forces the whole
# parquet scan + filter + sort to run here (mirroring blosc2's where() and
# arrow's to_table()), but -- unlike fetchdf() -- it never imports pandas, so
# its ~0.2 s lazy-import cost no longer hides inside this "compute" timer.
con.sql(query).create("result")
t2 = time.perf_counter()

# DuckDB's native box renderer reads only the small materialized table (no
# re-scan of the parquet) and is also pandas-free.
con.table("result").show(max_rows=10)
t3 = time.perf_counter()

# Footer matching pandas' "[N rows x M columns]" so benchmark harnesses can do a
# cross-method row-count check without importing pandas here.
nrows = con.sql("SELECT count(*) FROM result").fetchone()[0]
ncols = len(con.table("result").columns)
print(f"[{nrows} rows x {ncols} columns]")

print(f"open:    {t1 - t0:.6f} s")
print(f"compute: {t2 - t1:.6f} s")
print(f"print:   {t3 - t2:.6f} s")
print(f"total:   {t3 - t0:.6f} s")

con.close()
