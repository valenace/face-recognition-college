"""
Clasificación de desconocidos para la asistencia en vivo.

El motor devuelve `DESCONOCIDO` en cada frame donde la similitud no alcanza el
umbral de identidad (0.30). Este módulo decide cuándo una pista desconocida
representa a una persona no registrada real (`INTRUSO`) y evita dos extremos:
1. Que un pico aislado de un solo frame (>= 0.18) descarte o borre a casi todos
   los desconocidos reales del salón.
2. Que una misma persona no registrada sentada en su pupitre genere múltiples
   filas duplicadas cuando ByteTrack pierde el rastro unos segundos y le asigna
   un `track_id` nuevo en el mismo asiento.

Este módulo es lógica pura (sin Django) para poder probarlo con `pytest -m unit`.
Lo usa `core.registro_asistencia.aplicar_decisiones`.
"""

import math

# Umbral sobre la similitud PROMEDIO de la pista para descartarla como posible
# estudiante registrado no confirmado (cuando el candidato no está ya anclado
# simultáneamente en otra pista activa del aula).
UMBRAL_INTRUSO = 0.24
# Umbral sobre la MEJOR similitud efectiva de la pista. Si en algún momento se
# acercó a un registrado, la pista no es un intruso aunque su promedio sea bajo:
# el promedio es acumulativo y no baja cuando la cara mejora.
UMBRAL_MEJOR = 0.26
# Frames consecutivos que debe sostener la pista para crear la fila.
FRAMES_INTRUSO = 15
# Frames sin ver la pista para limpiar su estado interno.
OLVIDO_PISTA = 30
# Ventana (en frames) y factor de distancia (en múltiplos del ancho del rostro)
# para reutilizar la misma fila de INTRUSO cuando ByteTrack reasigna track_id
# en el mismo asiento.
OLVIDO_ESPACIAL = 240
FACTOR_DISTANCIA_ASIENTO = 2.5


class SeguimientoIntrusos:
    """Decide cuándo un desconocido es un intruso real y deduplica por asiento.

    Reglas:
    - Se evalúa el PROMEDIO de similitud de la pista (`suma / muestras`), no el
      máximo de un único frame aislado.
    - Y además su MEJOR similitud efectiva: si en algún momento la pista se
      acercó a un registrado, no es un intruso (el promedio es acumulativo y no
      refleja una mejora posterior de la cara).
    - Si el `candidato` más cercano ya está confirmado/anclado en OTRA pista
      activa en el mismo frame, esa muestra no penaliza a la pista desconocida
      (dos personas distintas en el mismo instante no pueden ser el mismo alumno).
    - Si la pista se confirma (`CONFIRMADO`/`ANCLADO`) o su promedio supera
      `umbral` (y fue la creadora de la fila), su fila se descarta.
    - Los conflictos de identidad nunca cuentan como intruso.
    - Se exigen `frames` frames consecutivos de la misma pista para calificar.
    - Si una nueva pista desconocida califica en la misma posición espacial
      donde había un intruso reciente (pista anterior ya no visible), reutiliza
      su `registro_id` en vez de duplicar la fila.
    """

    def __init__(
        self,
        umbral=UMBRAL_INTRUSO,
        umbral_mejor=UMBRAL_MEJOR,
        frames=FRAMES_INTRUSO,
        olvido=OLVIDO_PISTA,
        olvido_espacial=OLVIDO_ESPACIAL,
        factor_distancia=FACTOR_DISTANCIA_ASIENTO,
    ):
        self.umbral = umbral
        self.umbral_mejor = umbral_mejor
        self.frames = frames
        self.olvido = olvido
        self.olvido_espacial = olvido_espacial
        self.factor_distancia = factor_distancia
        self.frame_idx = 0
        self._pistas = {}
        self._asientos = {}  # registro_id -> {bbox, ultimo, track_id, max_ancho}

    def _buscar_asiento_cercano(self, bbox, vistos):
        """Busca si la caja cae en el asiento de un intruso perdido recientemente."""
        if not bbox or len(bbox) < 4:
            return None
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        ancho = max(1.0, float(bbox[2] - bbox[0]))

        mejor_id = None
        mejor_dist = float("inf")
        for reg_id, asiento in self._asientos.items():
            if asiento["track_id"] in vistos:
                continue
            if self.frame_idx - asiento["ultimo"] > self.olvido_espacial:
                continue
            sbbox = asiento.get("bbox")
            if not sbbox:
                continue
            scx = (sbbox[0] + sbbox[2]) / 2.0
            scy = (sbbox[1] + sbbox[3]) / 2.0
            sancho = max(1.0, float(sbbox[2] - sbbox[0]))
            dist = math.hypot(cx - scx, cy - scy)
            limite = max(ancho, sancho) * self.factor_distancia
            if dist <= limite and dist < mejor_dist:
                mejor_dist = dist
                mejor_id = reg_id
        return mejor_id

    def evaluar(self, decisiones):
        """Actualiza el estado y devuelve (crear, descartar).

        `crear` son las decisiones que ya califican como intruso (incluyendo
        `reutilizar_registro_id` si corresponde al mismo asiento); `descartar`
        son los ids de filas que dejaron de calificar (la pista se confirmó o
        superó el umbral promedio).
        """
        self.frame_idx += 1
        crear, descartar = [], []
        vistos = {
            d.get("track_id")
            for d in decisiones
            if d.get("track_id") is not None
        }
        confirmados_en_frame = {
            d.get("nombre")
            for d in decisiones
            if d.get("estado") in ("CONFIRMADO", "ANCLADO") and d.get("nombre")
        }

        for decision in decisiones:
            track_id = decision.get("track_id")
            if track_id is None:
                continue
            estado = decision.get("estado")
            info = self._pistas.get(track_id)

            if estado in ("CONFIRMADO", "ANCLADO"):
                if info and info["registro_id"] and info.get("es_creador", True):
                    descartar.append(info["registro_id"])
                    self._asientos.pop(info["registro_id"], None)
                if info is not None:
                    info["descartada"] = True
                    info["registro_id"] = None
                continue
            if estado != "DESCONOCIDO":
                continue

            motivo = decision.get("motivo") or ""
            similitud = decision.get("similitud")
            if motivo.startswith("conflicto") or similitud is None:
                continue

            bbox = decision.get("bbox")
            if info is None:
                info = {
                    "best": 0.0,
                    "best_ef": 0.0,
                    "suma": 0.0,
                    "muestras": 0,
                    "rachas": 0,
                    "ultimo": self.frame_idx,
                    "registro_id": None,
                    "es_creador": True,
                    "descartada": False,
                    "bbox": bbox,
                }
                self._pistas[track_id] = info

            info["ultimo"] = self.frame_idx
            info["rachas"] += 1
            if bbox is not None:
                info["bbox"] = bbox
            if similitud > info["best"]:
                info["best"] = similitud

            candidato = decision.get("candidato")
            # Si el candidato top-1 ya está confirmado/anclado en otra pista de
            # este mismo frame, esta pista no puede ser esa persona.
            sim_efectiva = 0.0 if (candidato and candidato in confirmados_en_frame) else similitud
            info["suma"] += sim_efectiva
            info["muestras"] += 1
            promedio = info["suma"] / info["muestras"]
            if sim_efectiva > info["best_ef"]:
                info["best_ef"] = sim_efectiva

            # Se descarta si el promedio es alto O si en algún momento se acercó a
            # un registrado: el promedio es acumulativo y no refleja una mejora
            # posterior de la cara.
            if promedio >= self.umbral or (
                self.umbral_mejor is not None and info["best_ef"] >= self.umbral_mejor
            ):
                info["descartada"] = True
                if info["registro_id"] and info.get("es_creador", True):
                    descartar.append(info["registro_id"])
                    self._asientos.pop(info["registro_id"], None)
                    info["registro_id"] = None
                continue

            if info["registro_id"] is not None:
                asiento = self._asientos.get(info["registro_id"])
                if asiento is not None:
                    asiento["ultimo"] = self.frame_idx
                    asiento["track_id"] = track_id
                    if bbox is not None:
                        asiento["bbox"] = bbox

            if (
                not info["descartada"]
                and info["rachas"] >= self.frames
                and info["registro_id"] is None
            ):
                reutilizar_id = self._buscar_asiento_cercano(bbox, vistos - {track_id})
                if reutilizar_id is not None:
                    decision["reutilizar_registro_id"] = reutilizar_id
                    asiento = self._asientos.get(reutilizar_id)
                    if bbox is not None and asiento is not None:
                        ancho = bbox[2] - bbox[0]
                        if ancho > asiento.get("max_ancho", 0):
                            decision["mejorar_captura"] = True
                crear.append(decision)

        for track_id, info in list(self._pistas.items()):
            if track_id not in vistos and self.frame_idx - info["ultimo"] > self.olvido:
                promedio = (
                    info["suma"] / info["muestras"] if info["muestras"] else 0.0
                )
                if (
                    info["registro_id"]
                    and info.get("es_creador", True)
                    and promedio >= self.umbral
                ):
                    descartar.append(info["registro_id"])
                    self._asientos.pop(info["registro_id"], None)
                del self._pistas[track_id]

        return crear, descartar

    def marcar_registrado(self, track_id, registro_id, bbox=None, es_reutilizado=False):
        """Asocia la fila creada o reutilizada a la pista y registra su asiento."""
        info = self._pistas.get(track_id)
        if info is not None:
            info["registro_id"] = registro_id
            info["es_creador"] = not es_reutilizado
            if bbox is None:
                bbox = info.get("bbox")
        ancho = (bbox[2] - bbox[0]) if (bbox and len(bbox) >= 4) else 0
        asiento = self._asientos.get(registro_id)
        if asiento is None:
            self._asientos[registro_id] = {
                "bbox": bbox,
                "ultimo": self.frame_idx,
                "track_id": track_id,
                "max_ancho": ancho,
            }
        else:
            asiento["ultimo"] = self.frame_idx
            asiento["track_id"] = track_id
            if bbox is not None:
                asiento["bbox"] = bbox
            if ancho > asiento.get("max_ancho", 0):
                asiento["max_ancho"] = ancho
