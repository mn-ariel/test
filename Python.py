# %pyspark
from pyspark.sql import functions as F
from pyspark import StorageLevel
from pyspark.sql import DataFrame

# ---- Configuración ligera
spark.conf.set("spark.sql.shuffle.partitions", "16")  # ajusta según tamaño
sc.setCheckpointDir("/tmp/ckpt")  # o HDFS/S3 si aplica

# ---- Funciones puras
def add_random_key(df: DataFrame, seed: int) -> DataFrame:
    # Add one random column to reuse across all samples
    return df.withColumn("_r", F.rand(seed))

def stratified_sample(df: DataFrame, by_col: str, fractions: dict, seed: int) -> DataFrame:
    # Uses built-in stratified sampling; no pandas, no collect
    return df.stat.sampleBy(by_col, fractions, seed)

def uniform_sample(df: DataFrame, frac: float, seed: int) -> DataFrame:
    # Reuse _r if exists to avoid extra computation
    col = "_r" if "_r" in df.columns else F.rand(seed)
    return df.where((F.col("_r") if "_r" in df.columns else col) < frac)

def lightweight_metrics(df: DataFrame) -> dict:
    # Only cheap actions; avoid full scans if possible
    # Example: approximate counts or agg with partials
    n = df.count()
    return {"rows": n}

def persist_disk(df: DataFrame) -> DataFrame:
    df.persist(StorageLevel.DISK_ONLY)
    return df

def save_parquet(df: DataFrame, path: str) -> None:
    (df
     .coalesce(1)          # baja particiones si hace sentido para salida
     .write.mode("overwrite")
     .parquet(path))

# ---- Orquestador
def run_pipeline(source_path: str, seed: int = 123, frac: float = 0.1) -> tuple[DataFrame, dict]:
    df = spark.read.parquet(source_path)

    df = add_random_key(df, seed)
    df = persist_disk(df)  # se reutiliza varias veces

    # Ejemplo de dos muestras reutilizando la misma columna aleatoria
    sample_a = uniform_sample(df, frac=frac, seed=seed)
    sample_b = uniform_sample(df, frac=frac/2, seed=seed+1)

    # Corta lineage si el plan se vuelve grande
    sample_a = sample_a.checkpoint(eager=True)

    # Métricas ligeras
    metrics_a = lightweight_metrics(sample_a)
    metrics_b = lightweight_metrics(sample_b)

    # Limpieza agresiva
    df.unpersist()

    # Guarda resultados (minimiza objetos vivos en driver)
    save_parquet(sample_a, "/tmp/out/sample_a")
    save_parquet(sample_b, "/tmp/out/sample_b")

    # Retorna un DF final pequeño (o ninguno) + JSON de resultados
    summary_json = {
        "seed": seed,
        "frac": frac,
        "sample_a": metrics_a,
        "sample_b": metrics_b
    }
    # Si necesitas un DF pequeño como resultado:
    result_df = spark.createDataFrame(
        [(summary_json["seed"], summary_json["frac"], summary_json["sample_a"]["rows"], summary_json["sample_b"]["rows"])],
        ["seed", "frac", "rows_a", "rows_b"]
    )
    return result_df, summary_json

# ---- Ejecutar
result_df, summary = run_pipeline("/data/input/my_table_parquet", seed=7, frac=0.08)
# Aquí puedes mostrar result_df.limit(20).show() en Zeppelin si necesitas ver algo rápido
