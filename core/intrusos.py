"""
Clasificación de desconocidos para la asistencia en vivo.

El motor devuelve `DESCONOCIDO` en cada frame donde la similitud no alcanza el
umbral de identidad. Un estudiante registrado que se mueve produce muchos de esos
frames y ByteTrack los reparte en pistas distintas; escribir una fila por pista
llenaba la sesión de intrusos falsos.

Este módulo es lógica pura (sin Django) para poder probarlo con `pytest -m unit`.
Lo usa `core.registro_asistencia.aplicar_decisiones`.
"""

# Umbral para decidir si un `DESCONOCIDO` es intruso de verdad. Se compara contra
# el MEJOR score que alcanza la pista en toda su vida, no contra el frame suelto:
# un estudiante registrado en un frame malo puede dar 0.10 en un frame y 0.28 en
# otro, mientras que un desconocido real no se acerca a los registrados.
# Calibrado con los videos 1 y 3 (docs/analisis/unc/11_intrusos.md):
#   video 1 (todos registrados): 130 pistas -> 9 filas con 0.18
#   video 3 (21 desconocidos): 18 de 21 pistas detectadas con 0.18
UMBRAL_INTRUSO = 0.18
# Frames consecutivos que debe sostener la pista para crear la fila.
FRAMES_INTRUSO = 15
# Frames sin ver la pista para considerarla desaparecida.
OLVIDO_PISTA = 30


class SeguimientoIntrusos:
    """Decide cuándo un desconocido es un intruso real y no un registrado perdido.

    Reglas:
    - Se sigue el MEJOR score de cada pista. Si en algún momento se acerca a los
      registrados (`>= umbral`), la pista deja de ser intruso y su fila se borra.
    - Los conflictos de identidad nunca cuentan como intruso.
    - Se exigen `frames` frames consecutivos de la misma pista para crear la fila.
    """

    def __init__(self, umbral=UMBRAL_INTRUSO, frames=FRAMES_INTRUSO, olvido=OLVIDO_PISTA):
        self.umbral = umbral
        self.frames = frames
        self.olvido = olvido
        self.frame_idx = 0
        self._pistas = {}

    def evaluar(self, decisiones):
        """Actualiza el estado y devuelve (crear, descartar).

        `crear` son las decisiones que ya califican como intruso; `descartar` son
        los ids de filas que dejaron de calificar (la pista mejoró).
        """
        self.frame_idx += 1
        crear, descartar = [], []
        vistos = set()

        for decision in decisiones:
            track_id = decision.get("track_id")
            if track_id is None:
                continue
            estado = decision.get("estado")
            info = self._pistas.get(track_id)

            if estado in ("CONFIRMADO", "ANCLADO"):
                if info and info["registro_id"]:
                    descartar.append(info["registro_id"])
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

            vistos.add(track_id)
            if info is None:
                info = {
                    "best": 0.0,
                    "rachas": 0,
                    "ultimo": self.frame_idx,
                    "registro_id": None,
                    "descartada": False,
                }
                self._pistas[track_id] = info
            info["ultimo"] = self.frame_idx
            info["rachas"] += 1
            if similitud > info["best"]:
                info["best"] = similitud

            if info["best"] >= self.umbral:
                # Se parece demasiado a un registrado: no es intruso.
                info["descartada"] = True
                if info["registro_id"]:
                    descartar.append(info["registro_id"])
                    info["registro_id"] = None
                continue

            if (
                not info["descartada"]
                and info["rachas"] >= self.frames
                and info["registro_id"] is None
            ):
                crear.append(decision)

        for track_id, info in list(self._pistas.items()):
            if track_id not in vistos and self.frame_idx - info["ultimo"] > self.olvido:
                # La pista desapareció: si había creado fila y seguía bajo el
                # umbral, se queda; si no, se descarta.
                if info["registro_id"] and info["best"] >= self.umbral:
                    descartar.append(info["registro_id"])
                del self._pistas[track_id]

        return crear, descartar

    def marcar_registrado(self, track_id, registro_id):
        """Asocia la fila creada a la pista para poder descartarla si mejora."""
        info = self._pistas.get(track_id)
        if info is not None:
            info["registro_id"] = registro_id
