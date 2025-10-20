# main.py
from __future__ import annotations
import logging
from logging.config import dictConfig
import pandas as pd

from factories import MuestreoFactory, MuestreoConfig  # tu factory mejorada

def setup_logging() -> None:
    dictConfig({
        "version": 1,
        "formatters": {"std": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"}},
        "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "std", "level": "INFO"}},
        "root": {"level": "INFO", "handlers": ["console"]},
    })

def run_pipeline(df: pd.DataFrame) -> dict:
    setup_logging()

    cfg = MuestreoConfig(
        df=df,
        identificador_df="folio",
        variables_control_continuas=["peso", "volumen"],
        variables_control_categoricas=["origen", "destino"],
        sup_proporcion=0.5,
        nivel_confianza=1.96,
        error_relativo=0.05,
        margen_muestra_extra=0.10,
        seed_global=42,
        cantidad_semillas=5,
    )

    muestreo = MuestreoFactory.crear_muestreo_homogeneo(cfg)

    logger = logging.getLogger("pipeline")

    # --- 1) tamaños de muestra
    muestreo.tamano_muestra_infinita, muestreo.calculo_muestra_infinita = \
        muestreo.calcular_tamano_muestra_poblacion_infinita(
            muestreo._p, muestreo._nc, muestreo._e_pct
        )

    muestreo.tamano_muestra_finita, muestreo.calculo_muestra_finita, muestreo.tamano_muestra_finita_ajustada = \
        muestreo.calcular_tamano_muestra_poblacion_finita(
            muestreo._df, muestreo._p, muestreo._nc, muestreo._e_pct, muestreo._m_m_x
        )

    # --- 2) generar y evaluar muestras
    (
        muestreo.resultado_mejor_grupo_homogeneo,
        estado_generacion,
        errores_generacion,
        muestreo.df_piloto,
        muestreo.df_control,
    ) = muestreo.generar_muestras_y_evaluar(
        df=muestreo._df,
        ident=muestreo._ident,
        ind_cont=muestreo._ind_cont,
        ind_cat=muestreo._ind_cat,
        muestra_minima_extra=muestreo.tamano_muestra_finita_ajustada,
        proporciones=muestreo._proportions,
        seeds=muestreo._seeds,
        p_threshold=muestreo._p_threshold,
        evaluar_variables_continuas=muestreo.evaluar_variables_continuas,   # si tus firmas lo requieren
        evaluar_variables_categoricas=muestreo.evaluar_variables_categoricas,
        seleccionar_mejor_grupo_homogeneo=muestreo.seleccionar_mejor_grupo_homogeneo
    )

    if estado_generacion == "ERROR":
        muestreo.estado = "ERROR"
        muestreo.errores.extend(errores_generacion)
        # si quieres abortar:
        raise RuntimeError(f"Error en generación de muestras: {errores_generacion}")

    # --- 3) post-proceso: mejor semilla
    if muestreo.resultado_mejor_grupo_homogeneo is not None:
        muestreo.mejor_semilla = muestreo.resultado_mejor_grupo_homogeneo.get("mejor_semilla")

    # --- 4) resumen de variables por grupo
    muestreo.resumen_variables_control = muestreo.analizar_datos_por_grupo(
        muestreo.df_piloto, muestreo.df_control, muestreo._ind_cont, muestreo._ind_cat
    )

    # --- 5) preparar resultados y guardar
    datos = muestreo.preparar_datos_resultado()
    muestreo.guardar_resultados_json("outputs/resultados.json")

    logger.info("Pipeline OK | tam_f=%s tam_f_adj=%s semilla=%s",
                muestreo.tamano_muestra_finita,
                muestreo.tamano_muestra_finita_ajustada,
                getattr(muestreo, "mejor_semilla", None))
    return datos

if __name__ == "__main__":
    # Carga tu DF real aquí
    df_demo = pd.DataFrame({"folio": [1,2,3], "peso":[10,12,9], "volumen":[1,1.2,0.9], "origen":["A","A","B"], "destino":["X","X","Y"]})
    run_pipeline(df_demo)
