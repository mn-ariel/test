v# core/muestreo_homogeneo.py
import logging
from typing import Any, Dict, Sequence, Tuple, Optional
import pandas as pd
from core.evaluadores import (
    generar_muestras_y_evaluar,
    analizar_datos_por_grupo,
)
from services.sizes import NormalApproxSize

logger = logging.getLogger(__name__)

class MuestreoHomogeneidad:
    def __init__(
        self,
        df: pd.DataFrame,
        identificador_df: str,
        variables_control_continuas: Sequence[str],
        variables_control_categoricas: Optional[Sequence[str]],
        sup_proporcion: float,
        nivel_confianza: float,
        error_relativo: float,
        proporciones_esperadas: Tuple[float, float],
        semillas_aleatorias: Sequence[int],
        umbral_homogeneidad: float,
        margen_muestra_extra: float,
        size_calc: Optional[NormalApproxSize] = None,
    ) -> None:
        self.df = df
        self.ident = identificador_df
        self.ind_cont = list(variables_control_continuas)
        self.ind_cat = list(variables_control_categoricas or [])
        self.p = sup_proporcion
        self.z = nivel_confianza
        self.e = error_relativo
        self.proporciones = proporciones_esperadas
        self.seeds = list(semillas_aleatorias)
        self.threshold = umbral_homogeneidad
        self.m_extra = margen_muestra_extra
        self.size_calc = size_calc or NormalApproxSize()
        self.estado = "OK"
        self.errores: list[str] = []
        self._resultado: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        try:
            n_total = len(self.df)
            n_inf = self.size_calc.sample_size_infinite(self.p, self.z, self.e)
            n_fin = self.size_calc.sample_size_finite(n_total, self.p, self.z, self.e)
            n_ajustada = int(n_fin * (1 + self.m_extra))

            resumen_control = analizar_datos_por_grupo(
                self.df, self.ind_cont, self.ind_cat
            )

            resultado, estado, errores, df_piloto, df_control = generar_muestras_y_evaluar(
                df=self.df,
                ident=self.ident,
                ind_cont=self.ind_cont,
                ind_cat=self.ind_cat,
                muestra_minima_extra=n_ajustada,
                proporciones=self.proporciones,
                seeds=self.seeds,
                p_threshold=self.threshold,
            )

            self.estado = estado
            self.errores.extend(errores)

            self._resultado = {
                "tamano_infinita": n_inf,
                "tamano_finita": n_fin,
                "tamano_ajustada": n_ajustada,
                "resumen_variables_control": resumen_control,
                "resultado_mejor_grupo": resultado,
                "df_piloto": df_piloto,
                "df_control": df_control,
                "seed_utilizada": resultado.get("mejor_semilla") if resultado else None,
            }
            return self._resultado
        except Exception as exc:
            self.estado = "ERROR"
            self.errores.append(str(exc))
            logger.exception("Error en run()")
            return {}

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self._resultado)
        if "df_piloto" in d:
            d["df_piloto"] = None  # exporta con Exporter, no aquí
        if "df_control" in d:
            d["df_control"] = None
        d["estado"] = self.estado
        d["errores"] = list(self.errores)
        return d
