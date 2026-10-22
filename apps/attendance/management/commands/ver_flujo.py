"""
Muestra paso a paso cómo el motor consulta la base vectorial y decide la asistencia.

Ejemplos:
    python manage.py ver_flujo --hasta 10 --detalle
    python manage.py ver_flujo --crear-sesion --detalle --sql
    python manage.py ver_flujo --sesion 12 --desde 30 --hasta 50 --mostrar

Sin `--sesion` ni `--crear-sesion` solo visualiza: no escribe en la base.
"""

import os
import time

import cv2
import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import connection

from apps.academic.models import (
    AsignacionClase,
    Asignatura,
    Estudiante,
    Inscripcion,
    Salon,
    Seccion,
)
from apps.attendance.models import RegistroAsistencia, SesionClase
from core.db_config import alias_de
from core.registro_asistencia import aplicar_decisiones, mapa_estudiantes

VIDEO_POR_DEFECTO = "data/unc/salon/video5096333890499381162.mp4"
COLORES = {
    "CONFIRMADO": (0, 255, 0),
    "ANCLADO": (0, 255, 0),
    "VERIFICANDO": (0, 165, 255),
    "CENTRATE": (128, 128, 128),
    "ERROR": (0, 165, 255),
    "DESCONOCIDO": (0, 0, 255),
    "FRAUDE": (0, 0, 255),
}


class Command(BaseCommand):
    help = "Muestra el flujo completo: consulta a pgvector, FAISS, decisiones y escritura en la base"

    def add_arguments(self, parser):
        parser.add_argument("--video", default=VIDEO_POR_DEFECTO)
        parser.add_argument("--desde", type=float, default=0.0)
        parser.add_argument("--hasta", type=float, default=10.0)
        parser.add_argument("--detalle", action="store_true", help="Una línea por cara detectada")
        parser.add_argument("--sql", action="store_true", help="Mostrar el SQL que escribe la asistencia")
        parser.add_argument("--mostrar", action="store_true", help="Abrir ventana con el video anotado")
        parser.add_argument("--cada", type=int, default=30, help="Resumen cada N frames")
        parser.add_argument("--sesion", type=int, help="ID de una sesión existente")
        parser.add_argument("--crear-sesion", action="store_true", help="Crear una sesión de demo")
        parser.add_argument("--limpiar", action="store_true", help="Borrar la sesión de demo al terminar")

    # ── Utilidades ────────────────────────────────────────────────
    def _linea(self, texto=""):
        self.stdout.write(texto)

    def _seccion(self, titulo):
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"=== {titulo} ==="))

    def _crear_demo(self):
        from django.contrib.auth import get_user_model

        User = get_user_model()
        profesor, _ = User.objects.get_or_create(
            username="prof_flujo",
            defaults={"first_name": "Profesor", "last_name": "Flujo",
                      "role": User.Role.PROFESSOR},
        )
        asignatura, _ = Asignatura.objects.get_or_create(
            codigo="DEMO-FLUJO", defaults={"nombre": "Demo de flujo"}
        )
        salon, _ = Salon.objects.get_or_create(
            nombre="Salón Demo", defaults={"capacidad": 30}
        )
        seccion, _ = Seccion.objects.get_or_create(codigo="DEMO-S")
        asignacion, _ = AsignacionClase.objects.get_or_create(
            profesor=profesor, asignatura=asignatura, seccion=seccion, salon=salon,
            defaults={"dia_semana": AsignacionClase.DiaSemana.LUNES,
                      "horario_inicio": "08:00", "horario_fin": "10:00"},
        )
        for estudiante in Estudiante.objects.all():
            Inscripcion.objects.get_or_create(
                estudiante=estudiante, asignacion_clase=asignacion
            )
        sesion, _ = SesionClase.objects.get_or_create(asignacion_clase=asignacion)
        RegistroAsistencia.objects.filter(sesion=sesion).delete()
        return sesion

    def _limpiar_demo(self, sesion):
        asignacion = sesion.asignacion_clase
        sesion.delete()
        Inscripcion.objects.filter(asignacion_clase=asignacion).delete()
        asignacion.delete()
        Asignatura.objects.filter(codigo="DEMO-FLUJO").delete()
        Salon.objects.filter(nombre="Salón Demo").delete()
        Seccion.objects.filter(codigo="DEMO-S").delete()
        from django.contrib.auth import get_user_model
        get_user_model().objects.filter(username="prof_flujo").delete()

    def _dibujar(self, frame, decisiones):
        for d in decisiones:
            x1, y1, x2, y2 = d["bbox"]
            color = COLORES.get(d["estado"], (0, 0, 255))
            if d.get("revision"):
                color = (0, 215, 255)  # amarillo: identidad posible, revisar
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            texto = f"posible {d['nombre']}" if d.get("revision") else d["nombre"]
            if d.get("similitud") is not None:
                texto += f" {d['similitud']:.2f}"
            cv2.putText(frame, texto, (x1 + 2, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    # ── Comando ───────────────────────────────────────────────────
    def handle(self, *args, **opciones):
        from scripts.asistencia_unificada_optimizada import MotorAsistencia
        from core.db_config import CONEXION, cargar_centroides
        from core.utils_facial import embedding_facial

        video = opciones["video"]
        desde, hasta = opciones["desde"], opciones["hasta"]
        if not os.path.exists(video):
            self.stderr.write(f"No existe el video: {video}")
            return

        # 1. Base vectorial
        self._seccion("1. CONSULTA A LA BASE VECTORIAL")
        with connection.cursor() as cur:
            cur.execute("select version()")
            self._linea(f"Servidor: {cur.fetchone()[0].split(',')[0]}")
            cur.execute("select extversion from pg_extension where extname='vector'")
            fila = cur.fetchone()
            self._linea(f"pgvector: {fila[0] if fila else 'no instalado'}")
        self._linea("")
        self._linea("SQL que ejecuta core/db_config.py::cargar_centroides():")
        self._linea("  SELECT u.username, u.first_name, e.embedding::text")
        self._linea("  FROM attendance_faceembedding e")
        self._linea("  JOIN users_user u ON u.id = e.user_id")
        self._linea("  ORDER BY u.username;")
        self._linea("")

        inicio = time.time()
        centroides = cargar_centroides()
        ms = (time.time() - inicio) * 1000
        self._linea(f"Filas: {len(centroides)}  (consulta + parseo: {ms:.1f} ms)")
        with connection.cursor() as cur:
            cur.execute("""select u.username, u.first_name, u.last_name, e.modelo,
                                  e.n_muestras, e.fuente
                           from attendance_faceembedding e
                           join users_user u on u.id = e.user_id order by u.username""")
            for usuario, nombre, apellido, modelo, muestras, fuente in cur.fetchall():
                self._linea(f"  {usuario:<12} {nombre} {apellido:<12} "
                            f"{modelo} · {muestras} muestras · {fuente}")

        # 2. Índice FAISS
        self._seccion("2. ÍNDICE EN MEMORIA")
        motor = MotorAsistencia(csv_path=os.path.join(settings.MEDIA_ROOT, "ver_flujo.csv"))
        motor.tamano_minimo_rostro = 20
        self._linea(f"faiss.IndexFlatIP(512) con {motor.indice_faiss.ntotal} vectores")
        self._linea("producto interno de vectores normalizados = similitud coseno")

        # 3. Comparación FAISS vs SQL con una cara real
        self._seccion("3. LA MISMA CARA EN FAISS Y EN SQL")
        captura = cv2.VideoCapture(video)
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(desde * (captura.get(cv2.CAP_PROP_FPS) or 30)))
        ok, frame_muestra = captura.read()
        captura.release()
        if ok:
            rostros = [r for r in motor.detector.detect(frame_muestra)
                       if (r.bbox[2] - r.bbox[0]) > motor.tamano_minimo_rostro
                       and float(r.confidence) >= motor.confianza_minima_deteccion]
            if rostros:
                rostro = max(rostros, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
                vec = embedding_facial(motor.recognizer, frame_muestra, rostro.landmarks)
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                self._linea(f"Cara de muestra: {int(rostro.bbox[2] - rostro.bbox[0])} px de ancho, "
                            f"confianza {float(rostro.confidence):.3f}")
                self._linea(f"Embedding: 512 dimensiones, norma {np.linalg.norm(vec):.3f}")
                self._linea("")
                self._linea("FAISS (en memoria):")
                dist, idx = motor.indice_faiss.search(vec.reshape(1, -1).astype("float32"), 3)
                for puesto in range(3):
                    self._linea(f"  {puesto + 1}. {motor.nombres_lista[idx[0][puesto]]:<10} "
                                f"{float(dist[0][puesto]):.4f}")
                self._linea("")
                self._linea("SQL equivalente (pgvector, operador <=> = distancia coseno):")
                with connection.cursor() as cur:
                    cur.execute(
                        """select u.username, 1 - (e.embedding <=> %s::vector) as sim
                           from attendance_faceembedding e
                           join users_user u on u.id = e.user_id
                           order by e.embedding <=> %s::vector limit 3""",
                        (vec.tolist(), vec.tolist()),
                    )
                    for puesto, (usuario, sim) in enumerate(cur.fetchall(), 1):
                        self._linea(f"  {puesto}. {alias_de(usuario):<10} {float(sim):.4f}")
            else:
                self._linea("No se detectó ninguna cara usable en ese frame.")

        # 4. Sesión de trabajo
        sesion = None
        if opciones["sesion"]:
            sesion = SesionClase.objects.get(id=opciones["sesion"])
            RegistroAsistencia.objects.filter(sesion=sesion).delete()
        elif opciones["crear_sesion"]:
            sesion = self._crear_demo()
            self._linea(self.style.SUCCESS(
                f"\nSesión de demo creada: id={sesion.id} "
                f"({sesion.asignacion_clase.asignatura.nombre})"
            ))
        else:
            self._linea("\nModo visualización: no se escribe en la base "
                        "(usa --crear-sesion o --sesion).")

        estudiantes = mapa_estudiantes(sesion.asignacion_clase) if sesion else {}
        profesor = sesion.asignacion_clase.profesor if sesion else None
        alias_profesor = alias_de(profesor.username, profesor.first_name) if profesor else None

        # 5. Procesamiento
        self._seccion("4. PROCESAMIENTO FRAME A FRAME")
        self._linea(f"umbral={motor.umbral_similitud}  votos={motor.votos_requeridos}  "
                    f"filtro={motor.tamano_minimo_rostro}px  conf={motor.confianza_minima_deteccion}  "
                    f"liveness<{motor.tamano_minimo_liveness}px omitido  ancla cada {motor.revalidar_cada}")
        if estudiantes:
            self._linea(f"Inscritos en la sesión: {', '.join(sorted(estudiantes))}")
        self._linea("")

        captura = cv2.VideoCapture(video)
        fps = captura.get(cv2.CAP_PROP_FPS) or 30
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(desde * fps))
        total = int(max(0, hasta - desde) * fps)
        conteo = {"frames": 0, "caras": 0, "confirmados": set(), "intrusos": 0, "fraudes": 0}
        sql_mostrado = 0
        connection.force_debug_cursor = True

        try:
            for indice in range(total):
                ok, frame = captura.read()
                if not ok:
                    break
                segundo = desde + indice / fps
                decisiones = motor.procesar_frame(frame)
                conteo["frames"] += 1
                conteo["caras"] += len(decisiones)

                if opciones["detalle"] or indice % opciones["cada"] == 0:
                    self._linea(f"frame {indice + 1:5d} · s{segundo:6.2f} · {len(decisiones)} caras detectadas")
                    if opciones["detalle"]:
                        for d in decisiones:
                            x1, y1, x2, y2 = d["bbox"]
                            sim = d.get("similitud")
                            if sim is None:
                                self._linea(f"    #{d['track_id']:<4} {x2-x1:3d}x{y2-y1:<3d} "
                                            f"{d['estado']}")
                                continue
                            margen = d.get("margen")
                            motivo = d.get("motivo")
                            etiqueta = d["nombre"] if d["estado"] in ("VERIFICANDO", "CENTRATE") else d["estado"]
                            if d.get("revision"):
                                etiqueta += " (posible)"
                            sufijo = f"   [{motivo}]" if motivo else ""
                            self._linea(
                                f"    #{d['track_id']:<4} {x2-x1:3d}x{y2-y1:<3d} "
                                f"conf={d.get('confianza_det', 0):.2f} "
                                f"top1={d.get('candidato') or '—':<9} {sim:.3f} "
                                f"top2={d.get('nombre2') or '—':<9} {d.get('sim_top2') or 0:.3f} "
                                f"margen={'—' if margen is None else f'{margen:.3f}'} "
                                f"-> {etiqueta}{sufijo}"
                            )

                if sesion is not None:
                    antes = len(connection.queries)
                    resumen = aplicar_decisiones(
                        decisiones, frame, sesion, estudiantes,
                        alias_profesor=alias_profesor, profesor=profesor,
                    )
                    conteo["confirmados"].update(resumen["confirmados"])
                    conteo["intrusos"] += resumen["intrusos"]
                    conteo["fraudes"] += resumen["fraudes"]
                    if opciones["sql"]:
                        for consulta in connection.queries[antes:]:
                            sql = consulta["sql"]
                            if sql.startswith("SELECT") or "attendance_registroasistencia" not in sql:
                                continue
                            if sql_mostrado < 30:
                                self._linea(f"    SQL: {sql[:240]}")
                                sql_mostrado += 1

                if opciones["mostrar"]:
                    cv2.imshow("Flujo de asistencia", self._dibujar(frame, decisiones))
                    if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                        break
        finally:
            captura.release()
            connection.force_debug_cursor = False
            if opciones["mostrar"]:
                cv2.destroyAllWindows()

        # 6. Resultado
        self._seccion("5. RESULTADO")
        self._linea(f"Frames: {conteo['frames']} · caras: {conteo['caras']} · "
                    f"confirmados: {sorted(conteo['confirmados'])} · "
                    f"detecciones desconocidas: {conteo['intrusos']} · "
                    f"detecciones de fraude: {conteo['fraudes']}")

        if sesion is not None:
            self._linea("")
            self._linea("Filas en attendance_registroasistencia:")
            filas = (RegistroAsistencia.objects.filter(sesion=sesion)
                     .select_related("estudiante__user").order_by("tipo_evento", "id"))
            if not filas:
                self._linea("  (ninguna)")
            for registro in filas:
                nombre = (registro.estudiante.user.get_full_name()
                          if registro.estudiante else "—")
                self._linea(f"  {nombre:<22} {registro.tipo_evento:<17} "
                            f"entrada={registro.hora_entrada} visto={registro.ultima_vez_visto} "
                            f"sim={registro.similitud_ia} fraude={registro.es_fraude}")
            self._linea("")
            self._linea("Consulta equivalente:")
            self._linea(f"  SELECT * FROM attendance_registroasistencia WHERE sesion_id={sesion.id};")

            if opciones["limpiar"]:
                self._limpiar_demo(sesion)
                self._linea(self.style.WARNING("\nSesión de demo eliminada."))

