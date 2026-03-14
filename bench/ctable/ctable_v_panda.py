import time
import numpy as np
import blosc2
import pandas as pd
from pydantic import BaseModel, Field
from typing import Annotated
import psutil

# --- 1. Tu RowModel COMPLEJO ---
class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype

class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True

# --- 2. Parámetros ---
N = 10_000_000  # 1M filas
print(f"=== BENCHMARK: 1M Filas COMPLEJAS (Listas de Listas) ===\n")

# ==========================================
# 0. GENERAR DATOS (Lista de listas COMPLEJA)
# ==========================================
print("--- Generando 1M filas complejas ---")
t0 = time.time()
data_list = []
for i in range(N):
    data_list.append([
        i,                                    # id: int64
        complex(i*0.1, i*0.01),              # c_val: complex128
        10.0 + np.sin(i*0.001)*50,           # score: float64
        (i % 3 == 0)                         # active: bool
    ])
t_gen = time.time() - t0
print(f"Tiempo generación: {t_gen:.4f} s")
print(f"Lista ocupa: {len(data_list):,} filas\n")

# ==========================================
# 1. PANDAS: Lista compleja -> DataFrame
# ==========================================
print("--- 1. PANDAS (Creación) ---")
gc_pandas = psutil.Process().memory_info().rss / (1024**2)
t0 = time.time()

df = pd.DataFrame(data_list, columns=['id', 'c_val', 'score', 'active'])

t_pandas_create = time.time() - t0
gc_pandas_after = psutil.Process().memory_info().rss / (1024**2)
mem_pandas = gc_pandas_after - gc_pandas
print(f"Tiempo creación:  {t_pandas_create:.4f} s")
print(f"Memoria usada:    {mem_pandas:.2f} MB")

# Pandas head(1000)
t0 = time.time()
df_head = df.head(N)
t_pandas_head = time.time() - t0
print(f"Tiempo head(1000): {t_pandas_head:.6f} s\n")

# ==========================================
# 2. BLOSC2 Oficial: extend() con conversión
# ==========================================
print("--- 2. BLOSC2 Oficial (extend + conversión Pydantic) ---")
gc_blosc = psutil.Process().memory_info().rss / (1024**2)
t0 = time.time()

# ❌ Blosc2 oficial REQUIERE conversión a modelos
ctable = blosc2.CTable(RowModel, expected_size=N)
ctable.extend(data_list)

t_blosc_create = time.time() - t0
gc_blosc_after = psutil.Process().memory_info().rss / (1024**2)
mem_blosc = gc_blosc_after - gc_blosc
mem_compressed = sum(col.schunk.nbytes for col in ctable._cols.values()) / (1024**2)
print(f"Tiempo creación:  {t_blosc_create:.4f} s")
total_comprimido = sum(col.cbytes for col in ctable._cols.values()) + ctable._valid_rows.cbytes
total_sin_comprimir = sum(col.nbytes for col in ctable._cols.values()) + ctable._valid_rows.nbytes

print(f"Comprimido: {total_comprimido / 1024 ** 2:.2f} MB")
print(f"Sin comprimir: {total_sin_comprimir / 1024 ** 2:.2f} MB")
print(f"Ratio: {total_sin_comprimir/total_comprimido:.2}x")

t0 = time.time()
ctable_head = ctable.head(N)
t_blosc_head = time.time() - t0
print(f"Tiempo head(1000): {t_blosc_head:.6f} s\n")



# ==========================================
# 🏆 RESUMEN COMPLETO
# ==========================================
print("═" * 80)
print("🥇 BENCHMARK 1M FILAS COMPLEJAS (int64+complex128+float64+bool)")
print("═" * 80)
print(f"{'MÉTRICA':<22} {'PANDAS':>12} {'BLOsc2*':>10} {'TU CTable':>12}")
print(f"{'':<22} {'':>12} {'*+Pydantic':>10} {'¡Directo!':>12}")
print("-" * 80)
