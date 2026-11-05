"""
Traduce las decisiones del motor calibrado a filas de `RegistroAsistencia`.

Lo usan tanto el procesamiento por lotes (`core/procesador_batch.py`) como la
captura en vivo (`core/servicio_en_vivo.py`), para que ambos flujos decidan con
las mismas reglas y escriban la asistencia de la misma forma.

Reglas:
- Estudiante inscrito y confirmado → ASISTENCIA_NORMAL (una fila por sesión).
- Estudiante registrado no inscrito en la sección → OTRA_SECCION.
- Profesor asignado → se anota en `notas_profesor`.
- Desconocido de tamaño usable → INTRUSO (sin marcarlo como fraude).
- Anti-spoofing positivo → SPOOFING con `es_fraude=True`.
"""

import uuid

import cv2
from django.core.files.base import ContentFile
from django.utils import timezone

from apps.academic.models import Estudiante
from apps.attendance.models import RegistroAsistencia
from core.intrusos import UMBRAL_INTRUSO

TAMANO_MINIMO_INTRUSO = 40  # px de ancho para guardar una captura de intruso


def mapa_estudiantes(asignacion):
    """Devuelve {alias_del_motor: Estudiante} para los inscritos en la clase."""
    from core.db_config import alias_de

    mapa = {}
    for inscripcion in asignacion.inscripciones_clase.select_related("estudiante__user"):
        usuario = inscripcion.estudiante.user
        mapa[alias_de(usuario.username, usuario.first_name)] = inscripcion.estudiante
    return mapa


def _guardar_captura(registro, frame, bbox):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    alto, ancho = frame.shape[:2]
    recorte = frame[max(0, y1):min(alto, y2), max(0, x1):min(ancho, x2)]
    if recorte.size == 0:
        return
    ok, buffer = cv2.imencode(".jpg", recorte)
    if not ok:
        return
    registro.captura_fraude.save(
        f"captura_{uuid.uuid4().hex[:8]}.jpg",
        ContentFile(buffer.tobytes()),
        save=True,
    )


def _registrar_desconocido(sesion, frame, bbox, es_fraude, tipo_evento, track_id, nota,
                           candidato=None, similitud=None):
    """Una fila por pista desconocida (no una por frame).

    La nota de la fila incluye la mejor coincidencia para que el profesor pueda
    juzgar si el 'intruso' es en realidad un estudiante registrado. El filtro para
    decidir si la pista califica vive en `SeguimientoIntrusos`.
    """
    etiqueta = f"{nota} Pista {track_id}."
    ahora = timezone.localtime().time()
    registro = RegistroAsistencia.objects.filter(
        sesion=sesion,
        estudiante=None,
        tipo_evento=tipo_evento,
        notas_auditoria__startswith=etiqueta,
    ).first()

    if registro is None:
        detalle = etiqueta
        if candidato:
            detalle += f" Mejor coincidencia: {candidato}"
            if similitud is not None:
                detalle += f" ({similitud:.2f})"
            detalle += "."
        registro = RegistroAsistencia.objects.create(
            sesion=sesion,
            estudiante=None,
            tipo_evento=tipo_evento,
            notas_auditoria=detalle,
            hora_entrada=ahora,
            ultima_vez_visto=ahora,
            es_fraude=es_fraude,
            similitud_ia=similitud,
        )
        _guardar_captura(registro, frame, bbox)
    else:
        registro.ultima_vez_visto = ahora
        if similitud is not None:
            registro.similitud_ia = similitud
        registro.save(update_fields=["ultima_vez_visto", "similitud_ia"])
    return registro


def aplicar_decisiones(decisiones, frame, sesion, estudiantes_por_alias,
                       alias_profesor=None, profesor=None, seguimiento=None):
    """Escribe en la base las decisiones de un frame/imagen.

    Con `seguimiento` (SeguimientoIntrusos) solo escribe los desconocidos que
    califican como intrusos y borra las filas de pistas que luego mejoraron.
    Sin él mantiene el comportamiento anterior (una fila por pista).

    Devuelve un resumen {confirmados, intrusos, fraudes} para mostrar en pantalla.
    """
    ahora = timezone.localtime().time()
    resumen = {"confirmados": [], "intrusos": 0, "fraudes": 0}

    intrusos_permitidos = None
    if seguimiento is not None:
        permitidos, descartar = seguimiento.evaluar(decisiones)
        intrusos_permitidos = {d["track_id"] for d in permitidos}
        if descartar:
            for registro in RegistroAsistencia.objects.filter(id__in=descartar):
                if registro.captura_fraude:
                    registro.captura_fraude.delete(save=False)
                registro.delete()

    for decision in decisiones:
        estado = decision["estado"]
        nombre = decision["nombre"]
        bbox = decision["bbox"]
        similitud = decision.get("similitud")
        track_id = decision.get("track_id")
        confirmado = estado in ("CONFIRMADO", "ANCLADO")

        if confirmado and nombre in estudiantes_por_alias:
            estudiante = estudiantes_por_alias[nombre]
            registro, creado = RegistroAsistencia.objects.get_or_create(
                sesion=sesion,
                estudiante=estudiante,
                defaults={
                    "hora_entrada": ahora,
                    "ultima_vez_visto": ahora,
                    "similitud_ia": similitud,
                    "tipo_evento": RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL,
                },
            )
            if not creado:
                registro.ultima_vez_visto = ahora
                if similitud is not None:
                    registro.similitud_ia = similitud
                registro.save(update_fields=["ultima_vez_visto", "similitud_ia"])
            resumen["confirmados"].append(nombre)
            continue

        if confirmado and alias_profesor and nombre == alias_profesor:
            nota = f"- Profesor {profesor.get_full_name()} detectado en aula."
            if nota not in (sesion.notas_profesor or ""):
                sesion.notas_profesor = (sesion.notas_profesor or "") + "\n" + nota
                sesion.save(update_fields=["notas_profesor"])
            continue

        if confirmado:
            # Registrado, pero de otra sección.
            usuario = f"unc_{nombre}"
            estudiante = Estudiante.objects.filter(user__username=usuario).first()
            if estudiante is not None:
                RegistroAsistencia.objects.get_or_create(
                    sesion=sesion,
                    estudiante=estudiante,
                    defaults={
                        "hora_entrada": ahora,
                        "ultima_vez_visto": ahora,
                        "similitud_ia": similitud,
                        "tipo_evento": RegistroAsistencia.TipoEvento.OTRA_SECCION,
                        "notas_auditoria": "Estudiante registrado fuera de esta sección.",
                    },
                )
            continue

        if estado == "FRAUDE":
            candidato = decision.get("candidato")
            # Si el estudiante candidato ya tiene asistencia confirmada en esta sesión,
            # no generamos una falsa alarma de spoofing por un giro de cabeza o pose extrema.
            ya_confirmado = False
            if candidato and (candidato in estudiantes_por_alias or f"unc_{candidato}" in estudiantes_por_alias):
                est = estudiantes_por_alias.get(candidato) or estudiantes_por_alias.get(f"unc_{candidato}")
                if est and RegistroAsistencia.objects.filter(
                    sesion=sesion,
                    estudiante=est,
                    tipo_evento=RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL,
                ).exists():
                    ya_confirmado = True
            elif candidato and RegistroAsistencia.objects.filter(
                sesion=sesion,
                estudiante__user__username__in=[candidato, f"unc_{candidato}"],
                tipo_evento=RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL,
            ).exists():
                ya_confirmado = True

            if not ya_confirmado:
                _registrar_desconocido(
                    sesion, frame, bbox, es_fraude=True,
                    tipo_evento=RegistroAsistencia.TipoEvento.SPOOFING,
                    track_id=track_id, nota="Posible fraude (anti-spoofing).",
                    candidato=candidato, similitud=similitud,
                )
                resumen["fraudes"] += 1
            continue

        if estado == "DESCONOCIDO":
            if intrusos_permitidos is not None and track_id not in intrusos_permitidos:
                continue
            ancho = int(bbox[2]) - int(bbox[0])
            if ancho >= TAMANO_MINIMO_INTRUSO:
                reutilizar_id = decision.get("reutilizar_registro_id")
                mejorar_captura = decision.get("mejorar_captura", False)
                registro = None
                if reutilizar_id is not None:
                    registro = RegistroAsistencia.objects.filter(
                        id=reutilizar_id, sesion=sesion
                    ).first()
                    if registro is not None:
                        registro.ultima_vez_visto = ahora
                        if similitud is not None:
                            registro.similitud_ia = similitud
                        registro.save(update_fields=["ultima_vez_visto", "similitud_ia"])
                        if mejorar_captura:
                            if registro.captura_fraude:
                                registro.captura_fraude.delete(save=False)
                            _guardar_captura(registro, frame, bbox)
                        if seguimiento is not None:
                            seguimiento.marcar_registrado(
                                track_id, registro.id, bbox=bbox, es_reutilizado=True
                            )
                if registro is None:
                    registro = _registrar_desconocido(
                        sesion, frame, bbox, es_fraude=False,
                        tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO,
                        track_id=track_id, nota="Persona no registrada.",
                        candidato=decision.get("candidato"), similitud=similitud,
                    )
                    if seguimiento is not None and registro is not None:
                        seguimiento.marcar_registrado(
                            track_id, registro.id, bbox=bbox, es_reutilizado=False
                        )
                resumen["intrusos"] += 1

    return resumen


# ──────────────────────────────────────────────────────────────
# Procesamiento por lotes: cada imagen es una observación independiente
# ──────────────────────────────────────────────────────────────

def reiniciar_pistas(motor):
    """Deja el motor listo para una imagen nueva sin recargar los modelos."""
    import supervision as sv

    motor.frame_idx = 0
    motor.identidades_ancladas = {}
    motor.votos_identidad = {}
    motor.votos_fraude = {}
    motor.vetos_identidad = {}
    motor.tracker = sv.ByteTrack()


def nuevo_acumulado():
    return {
        "confirmados": {},   # alias -> {n, sim_max, frame, bbox, primer_indice}
        "intrusos": {"n": 0, "frame": None, "bbox": None, "primer_indice": None},
        "fraudes": {"n": 0, "frame": None, "bbox": None, "primer_indice": None},
    }


def acumular_decisiones(acumulado, decisiones, frame, indice):
    """Acumula la evidencia de una imagen para decidir al final del lote."""
    for decision in decisiones:
        estado = decision["estado"]
        nombre = decision["nombre"]
        bbox = decision["bbox"]
        confirmado = estado in ("CONFIRMADO", "ANCLADO")

        if confirmado:
            info = acumulado["confirmados"].setdefault(
                nombre, {"n": 0, "sim_max": 0.0, "frame": frame, "bbox": bbox,
                         "primer_indice": indice}
            )
            info["n"] += 1
            sim = decision.get("similitud")
            if sim is not None and sim > info["sim_max"]:
                info["sim_max"] = sim
                info["frame"] = frame
                info["bbox"] = bbox
            continue

        if estado == "FRAUDE":
            info = acumulado["fraudes"]
        elif estado == "DESCONOCIDO":
            if int(bbox[2]) - int(bbox[0]) < TAMANO_MINIMO_INTRUSO:
                continue
            # En lote no hay tracking continuo: se filtra por la mejor coincidencia
            # y por los conflictos, que son registrados y no intrusos.
            similitud = decision.get("similitud")
            if similitud is not None and similitud >= UMBRAL_INTRUSO:
                continue
            if (decision.get("motivo") or "").startswith("conflicto"):
                continue
            info = acumulado["intrusos"]
        else:
            continue

        info["n"] += 1
        if info["frame"] is None:
            info["frame"] = frame
            info["bbox"] = bbox
            info["primer_indice"] = indice


def registrar_lote(acumulado, sesion, estudiantes_por_alias,
                   alias_profesor=None, profesor=None, minimo_imagenes=3):
    """Escribe en la base el resultado consolidado de un lote de imágenes.

    Un estudiante se registra si fue confirmado en al menos `minimo_imagenes`
    imágenes distintas (evidencia repetida, sin exigir que sean consecutivas).
    """
    ahora = timezone.localtime().time()
    registrados = []

    for alias, info in acumulado["confirmados"].items():
        if info["n"] < minimo_imagenes:
            continue

        if alias in estudiantes_por_alias:
            estudiante = estudiantes_por_alias[alias]
            RegistroAsistencia.objects.update_or_create(
                sesion=sesion,
                estudiante=estudiante,
                defaults={
                    "hora_entrada": ahora,
                    "ultima_vez_visto": ahora,
                    "similitud_ia": info["sim_max"],
                    "tipo_evento": RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL,
                },
            )
            registrados.append(alias)
            continue

        if alias_profesor and alias == alias_profesor:
            nota = f"- Profesor {profesor.get_full_name()} detectado en aula."
            if nota not in (sesion.notas_profesor or ""):
                sesion.notas_profesor = (sesion.notas_profesor or "") + "\n" + nota
                sesion.save(update_fields=["notas_profesor"])
            continue

        estudiante = Estudiante.objects.filter(user__username=f"unc_{alias}").first()
        if estudiante is not None:
            RegistroAsistencia.objects.update_or_create(
                sesion=sesion,
                estudiante=estudiante,
                defaults={
                    "hora_entrada": ahora,
                    "ultima_vez_visto": ahora,
                    "similitud_ia": info["sim_max"],
                    "tipo_evento": RegistroAsistencia.TipoEvento.OTRA_SECCION,
                    "notas_auditoria": "Estudiante registrado fuera de esta sección.",
                },
            )

    if acumulado["intrusos"]["n"]:
        info = acumulado["intrusos"]
        etiqueta = f"Personas no registradas en el lote ({info['n']} detecciones)."
        registro, creado = RegistroAsistencia.objects.get_or_create(
            sesion=sesion, estudiante=None,
            tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO,
            notas_auditoria=etiqueta,
            defaults={"hora_entrada": ahora, "ultima_vez_visto": ahora, "es_fraude": False},
        )
        if creado:
            _guardar_captura(registro, info["frame"], info["bbox"])

    if acumulado["fraudes"]["n"]:
        info = acumulado["fraudes"]
        etiqueta = f"Posible fraude en el lote ({info['n']} detecciones)."
        registro, creado = RegistroAsistencia.objects.get_or_create(
            sesion=sesion, estudiante=None,
            tipo_evento=RegistroAsistencia.TipoEvento.SPOOFING,
            notas_auditoria=etiqueta,
            defaults={"hora_entrada": ahora, "ultima_vez_visto": ahora, "es_fraude": True},
        )
        if creado:
            _guardar_captura(registro, info["frame"], info["bbox"])

    return registrados
