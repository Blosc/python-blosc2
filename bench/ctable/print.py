import time
import numpy as np
import pandas as pd
import blosc2
from pydantic import BaseModel, Field

# --- 1. Definir el Modelo ---
class RowModel(BaseModel):
    id: int = Field(ge=0)
    name: bytes = Field(default=b"unknown", max_length=10)
    score: float

# --- 2. Parámetros ---
N = 100_000
row_data = {"id": 1, "name": b"benchmark", "score": 3.14}

print(f"=== BENCHMARK: Ingestión Iterativa ({N} filas) ===\n")

# ==========================================
# TEST PANDAS (Baseline)
# ==========================================
print("--- 1. PANDAS (Lista -> DataFrame) ---")
t0 = time.time()

buffer_list = []
for _ in range(N):
    buffer_list.append(row_data)

df = pd.DataFrame(buffer_list)
t_pandas = time.time() - t0

print(f"Tiempo Total: {t_pandas:.4f} s")
mem_pandas = df.memory_usage(deep=True).sum() / (1024**2)
print(f"Memoria RAM:  {mem_pandas:.2f} MB")

print("\n--- PANDAS: Primeras 1000 líneas ---")
t0_print = time.time()
print(df.head(1000).to_string())
t_print_pandas = time.time() - t0_print
print(f"\nTiempo de impresión: {t_print_pandas:.4f} s")


# ==========================================
# TEST BLOSC2 (Estrategia: extend() con lista)
# ==========================================
print("\n" + "="*60)
print("--- 2. BLOSC2 (extend con lista de dicts) ---")
t0 = time.time()

# Acumular en lista de diccionarios
buffer_list_2 = []
for _ in range(N):
    buffer_list_2.append(row_data)

# Crear CTable vacía e insertar todo de golpe
ctable = blosc2.CTable(RowModel)
ctable.extend(buffer_list_2)

t_blosc_extend = time.time() - t0
print(f"Tiempo Total: {t_blosc_extend:.4f} s")

mem_blosc_extend = sum(col.schunk.nbytes for col in ctable._cols.values()) / (1024**2)
print(f"Memoria (Compr): {mem_blosc_extend:.2f} MB")

print("\n--- BLOSC2: Primeras 1000 líneas ---")
t0_print = time.time()
ctable_head = ctable.head(1000)
print(ctable_head)
t_print_blosc = time.time() - t0_print
print(f"\nTiempo de impresión: {t_print_blosc:.4f} s")

# ==========================================
# CONCLUSIONES
# ==========================================
print("\n" + "="*60)
print("--- RESUMEN ---")
print(f"Pandas (lista->df):       {t_pandas:.4f} s")
print(f"Blosc2 (extend):          {t_blosc_extend:.4f} s ({t_pandas/t_blosc_extend:.2f}x {'más rápido' if t_blosc_extend < t_pandas else 'más lento'})")
print(f"\nImpresión Pandas:         {t_print_pandas:.4f} s")
print(f"Impresión Blosc2:         {t_print_blosc:.4f} s")
print(f"\nCompresión Blosc2 vs Pandas: {mem_blosc_extend / mem_pandas * 100:.2f}% del tamaño")
