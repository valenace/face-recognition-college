# Motor de asistencia — Documentación técnica

## Capítulo 1 — `scripts/asistencia_unificada_optimizada.py`

**Archivo:** `scripts/asistencia_unificada_optimizada.py` · **595 líneas**
**Rol:** es el motor. Detecta rostros, los sigue, los compara contra los centroides
y decide identidad, intruso o fraude. No dibuja en los flujos web: devuelve una
lista de decisiones y cada consumidor decide qué hacer con ellas.

**Quién lo usa:**

| Consumidor | Cómo |
|---|---|
| `python scripts/asistencia_unificada_optimizada.py` | CLI con cámara o video |
| `core/servicio_en_vivo.py` | Hilo de la captura web `/en-vivo/` |
| `core/procesador_batch.py` | Procesamiento de una carpeta de imágenes |
| `apps/attendance/management/commands/ver_flujo.py` | Diagnóstico paso a paso |
| `experiments/unc/*` | Evaluaciones y demos |

**Dependencias externas:** `opencv-python` (cv2), `numpy`, `faiss-cpu`,
`supervision` (ByteTrack), `uniface` (RetinaFace, ArcFace, anti-spoofing).

---

## Mapa del archivo

| Líneas | Bloque | Qué hace |
|---|---|---|
| 1-17 | Docstring | Describe la arquitectura en 13 puntos |
| 19-36 | Imports | Librerías y `sys.path` |
| 38-46 | `COLORES` | Color de cada estado para la ventana |
| 49-96 | `_contexto_sesion()` | Carga la sesión de la BD (solo modo `--sesion`) |
| 99-202 | `MotorAsistencia.__init__()` | Carga centroides, arma FAISS y los modelos |
| 204-221 | `_preparar_archivo_csv()` | Crea/lee el CSV de experimentos |
| 223-249 | `_registrar_asistencia()` | Escribe una confirmación en el CSV |
| 251-260 | `_calcular_iou()` | Solapamiento entre dos cajas |
| 262-273 | `_contar_voto()` | Consenso temporal de 3 votos |
| 275-285 | `_evaluar_liveness()` | Anti-spoofing con umbral de tamaño |
| 287-336 | `_decidir_identidad()` | **El corazón de la decisión** |
| 338-351 | `_marcar_revision()` | Banda 0.30-0.40 para la UI |
| 353-392 | `_resolver_conflictos()` | Dos pistas no pueden ser la misma persona |
| 394-446 | `procesar_frame()` | Detección → tracking → decisión |
| 448-521 | `iniciar_vigilancia()` | Bucle de cámara/video y escritura |
| 523-536 | `_dibujar_ui()` | Cajas y etiquetas de la ventana |
| 542-595 | `__main__` | Argumentos de línea de comandos |

---

## 1. Docstring (líneas 1-17)

```python
"""
asistencia_unificada_optimizada.py — Motor de Inferencia Biométrico en Tiempo Real
================================================================================
Arquitectura:
  - Tracking: ByteTrack para mantener IDs consistentes en multitudes. Se actualiza
    también en frames sin detecciones para envejecer pistas perdidas.
  - Reconocimiento: centroide por estudiante + FAISS IndexFlatIP (similitud coseno).
  - Anti-spoofing: se evalúa en CADA decisión de reconocimiento, sin importar el tamaño
    del rostro. Error del spoofer != fraude (no suma votos ni confirma identidad).
  - Consenso temporal: 3 votos consecutivos (ventana de frames) del mismo candidato.
  - Ancla de identidad: se revalida cada REVALIDAR_CADA frames.
  - Margen top-2: se registra la segunda mejor coincidencia para análisis; la decisión
    todavía NO depende del margen.
  - Preprocesado compartido con el registro (core.utils_facial.embedding_facial).
  - Escritura: CSV opcional para experimentos, o base de datos (RegistroAsistencia)
    con las mismas reglas que la web cuando se indica una sesión (`--sesion`).
"""
```

- **Línea 2:** nombre del archivo y una frase que resume su función: inferencia
  biométrica en tiempo real.
- **Líneas 4-13:** la arquitectura en ocho decisiones de diseño. Cada línea describe
  un mecanismo que se explica más abajo con su código:
  - **ByteTrack** mantiene un `track_id` por persona entre frames, incluso cuando un
    frame no tiene detecciones, para "envejecer" las pistas perdidas.
  - **FAISS `IndexFlatIP`** busca el centroide más parecido con producto interno.
  - **Anti-spoofing** se evalúa en cada decisión; un error del modelo no es fraude.
  - **3 votos** consecutivos del mismo candidato antes de confirmar.
  - **Ancla**: una identidad confirmada se reutiliza 30 frames sin recalcular.
  - **Margen top-2** se guarda para análisis, pero **no decide** todavía.
  - **`embedding_facial`** es el mismo preprocesado que usa el registro: si cambia
    uno, cambia el otro, y las similitudes siguen siendo comparables.
- **Líneas 15-16:** la escritura tiene dos caminos: CSV (experimentos) o
  `RegistroAsistencia` (base de datos, con `--sesion`). Es la unificación que se
  hizo el 2026-10-22.

> **Por qué importa:** el docstring es el contrato del archivo. Cualquier cambio en
> el motor debe respetar estas reglas o actualizar el docstring; si no, el resto de
> la documentación (y los experimentos) quedan desincronizados.

---

## 2. Imports y `sys.path` (líneas 19-36)

```python
import cv2
import numpy as np
import pickle
import time
import csv
import faiss
from pathlib import Path
from datetime import datetime
import sys
import os

# Inyección al PYTHONPATH para encontrar la carpeta 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uniface import create_detector, create_recognizer
from uniface.spoofing import create_spoofer
import supervision as sv
from core.utils_facial import embedding_facial
```

- **Línea 19 `import cv2`:** OpenCV. Se usa para leer cámara/video, detectar,
  dibujar y codificar imágenes.
- **Línea 20 `import numpy as np`:** vectores de 512 dimensiones, normas,
  promedios y matrices. Todo el cálculo numérico pasa por aquí.
- **Línea 21 `import pickle`:** solo se usa si `usar_bd=False`; permite cargar el
  respaldo histórico `data/database_embeddings.pkl`.
- **Línea 22 `import time`:** medir FPS y respetar el ritmo del video.
- **Línea 23 `import csv`:** escribir el CSV de experimentos.
- **Línea 24 `import faiss`:** índice vectorial. `IndexFlatIP` hace producto
  interno exacto; como los vectores tienen norma 1, equivale a similitud coseno.
- **Línea 25 `from pathlib import Path`:** rutas de archivo multiplataforma.
- **Línea 26 `from datetime import datetime`:** fecha y hora de la asistencia.
- **Línea 27 `import sys` / línea 28 `import os`:** manipular `sys.path`.
- **Línea 31 `sys.path.append(...)`:** añade la raíz del proyecto (el padre de
  `scripts/`) al `PYTHONPATH`. Sin esto, `from core.utils_facial ...` fallaría al
  ejecutar el archivo directamente (`python scripts/...`). **No borrar**: el script
  debe funcionar tanto standalone como importado por Django.
- **Línea 33-34 `from uniface import ...`:** los tres modelos:
  - `create_detector("retinaface")` → detecta cajas y landmarks.
  - `create_recognizer("arcface")` → convierte una cara alineada en un vector de 512.
  - `create_spoofer()` → clasifica la cara como real o fraudulenta.
- **Línea 35 `import supervision as sv`:** `sv.ByteTrack` y `sv.Detections`.
- **Línea 36 `from core.utils_facial import embedding_facial`:** el preprocesado
  compartido con el registro. Es la pieza que garantiza que el vector del motor y
  el del registro se calculen igual.

---

## 3. Tabla de colores (líneas 38-46)

```python
COLORES = {
    "CONFIRMADO": (0, 255, 0),
    "ANCLADO": (0, 255, 0),
    "VERIFICANDO": (0, 165, 255),
    "CENTRATE": (0, 165, 255),
    "ERROR": (0, 165, 255),
    "DESCONOCIDO": (0, 0, 255),
    "FRAUDE": (0, 0, 255),
}
```

- **Líneas 39-40:** verde para identidad confirmada y anclada.
- **Líneas 41-43:** naranja para estados transitorios (`VERIFICANDO`, `CENTRATE`,
  `ERROR`). No son errores definitivos, solo avisos.
- **Línea 44:** rojo para `DESCONOCIDO`.
- **Línea 45:** rojo para `FRAUDE`.

Formato `(B, G, R)`: OpenCV usa el orden inverso a RGB. `(0, 255, 0)` es verde puro;
`(0, 165, 255)` es naranja; `(0, 0, 255)` es rojo.

La web no usa este diccionario: `core/servicio_en_vivo.py` y `ver_flujo.py` tienen
su propia copia con los mismos valores para no depender de un módulo que carga
modelos.

---

## 4. `_contexto_sesion()` (líneas 49-96)

```python
def _contexto_sesion(sesion_id):
    """Carga la sesión y sus inscritos para escribir en la base.

    Se importa Django aquí y no al inicio del módulo: el motor también se usa
    desde la terminal sin Django (inferencia pura y experimentos).
    """
    from core.sesion_demo import configurar_django

    configurar_django()

    from django.core.exceptions import ObjectDoesNotExist

    from apps.attendance.models import SesionClase
    from core.db_config import alias_de
    from core.registro_asistencia import mapa_estudiantes

    try:
        sesion = SesionClase.objects.select_related("asignacion_clase__profesor").get(
            id=sesion_id
        )
    except ObjectDoesNotExist:
        abiertas = (
            SesionClase.objects.filter(estado=SesionClase.Estado.EN_CURSO)
            .select_related("asignacion_clase__asignatura")
            .order_by("id")
        )
        if abiertas:
            detalle = "; ".join(
                f"{s.id} = {s.asignacion_clase.asignatura.nombre} ({s.fecha})"
                for s in abiertas
            )
            raise ValueError(
                f"No existe una sesión con id={sesion_id}. "
                f"Sesiones en curso: {detalle}."
            ) from None
        raise ValueError(
            f"No existe una sesión con id={sesion_id} y no hay sesiones en curso. "
            "Abre una desde el panel del profesor o créala en /admin/."
        ) from None

    asignacion = sesion.asignacion_clase
    profesor = asignacion.profesor
    return (
        sesion,
        mapa_estudiantes(asignacion),
        profesor,
        alias_de(profesor.username, profesor.first_name),
    )
```

Esta función solo se ejecuta en modo `--sesion`. Su trabajo es traducir un
`sesion_id` a todo lo que la capa de escritura necesita.

- **Línea 55-57:** importa `configurar_django` y lo ejecuta. El import es
  **perezoso** a propósito: cargar Django al inicio rompería la inferencia pura
  (experimentos sin base de datos). `configurar_django()` hace
  `DJANGO_SETTINGS_MODULE=cv_college.settings` y `django.setup()` una sola vez.
- **Línea 59:** `ObjectDoesNotExist` es la excepción base de Django; sirve para
  capturar `SesionClase.DoesNotExist` sin importar el modelo concreto.
- **Línea 61:** `SesionClase` es el modelo que representa una captura concreta.
- **Línea 62:** `alias_de` convierte `unc_alexa` en `alexa`, la clave que usa el
  motor y la que guarda FAISS.
- **Línea 63:** `mapa_estudiantes` devuelve `{alias: Estudiante}` solo con los
  inscritos en la asignación de esa sesión.
- **Línea 66-68:** busca la sesión por id y trae de una sola consulta la asignación
  y el profesor (`select_related` evita el N+1).
- **Línea 69:** si no existe, entra al manejo de error.
- **Línea 70-74:** busca las sesiones que están `EN_CURSO`, con su asignatura, para
  ofrecerlas como alternativa.
- **Línea 75-79:** si hay alguna, arma un texto `id = asignatura (fecha); ...`
  para que el usuario sepa qué ids puede usar.
- **Línea 80-83:** lanza `ValueError` con ese texto. `from None` oculta la
  excepción original y deja un mensaje limpio en la terminal.
- **Línea 84-87:** si no hay ninguna sesión abierta, el mensaje explica cómo
  crearla.
- **Línea 89:** `asignacion` es la clase (profesor + asignatura + sección + salón).
- **Línea 90:** `profesor` se usa para anotar su presencia en `notas_profesor`.
- **Línea 91-96:** devuelve una tupla de cuatro elementos:
  1. la sesión,
  2. el mapa de inscritos,
  3. el profesor,
  4. su alias (para detectar si el profesor aparece en cámara).

> **Qué NO hace:** no crea la sesión ni modifica datos. Solo lee y valida. La
> creación vive en `core/sesion_demo.py` (modo demo) o en el panel del profesor.

---

## 5. `MotorAsistencia.__init__()` (líneas 99-202)

### 5.1 Firma y parámetros (líneas 100-113)

```python
class MotorAsistencia:
    def __init__(
        self,
        db_path="data/database_embeddings.pkl",
        csv_path="data/asistencia.csv",
        usar_bd=True,
        registrar_csv=True,
    ):
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.usar_bd = usar_bd
        # La base de datos es la fuente de verdad. El CSV se conserva para los
        # experimentos y scripts que lo piden explícitamente; los flujos web y el
        # modo --sesion lo desactivan (ver core/registro_asistencia.py).
        self.registrar_csv = registrar_csv
```

- **Línea 99:** la clase. Todo el motor es un objeto con estado: modelos, índice,
  tracker y memorias de votos.
- **Línea 102 `db_path`:** ruta del pickle. Solo se usa cuando `usar_bd=False`, es
  decir, en experimentos que quieren reproducir el pasado sin tocar PostgreSQL.
- **Línea 103 `csv_path`:** ruta del CSV de experimentos. El modo base de datos lo
  ignora.
- **Línea 104 `usar_bd`:** `True` (por defecto) carga los centroides desde
  pgvector, la fuente de verdad. `False` cae al pickle.
- **Línea 105 `registrar_csv`:** `True` (por defecto) conserva el comportamiento de
  los experimentos: escribir en CSV. Los flujos web y el modo `--sesion` pasan
  `False` para que la única escritura sea `RegistroAsistencia`.
- **Línea 107-108:** convierte las rutas a `Path`. Así `csv_path.parent` funciona
  aunque llegue un string.
- **Línea 109:** guarda la bandera de origen de centroides.
- **Línea 113:** guarda la bandera de escritura CSV.

> **Por qué el default es `True`:** los scripts de `experiments/` y los tests del
> CSV dependen de ese comportamiento. Cambiarlo rompería la reproducibilidad de
> los resultados ya publicados. La web opta explícitamente por `False`.

### 5.2 Parámetros de decisión (líneas 115-146)

```python
        # Calibrado en Fase 4 con decisión de 3 frames: 0.30 da precisión 1.000
        # y 0 falsos positivos sobre la muestra etiquetada (docs/analisis/unc/04_resultados_curva_roc.md §7).
        self.umbral_similitud = 0.30
        self.umbral_liveness = 0.90
        self.tamano_minimo_rostro = 40
        # Confianza mínima del detector para procesar una cara. RetinaFace genera
        # falsos positivos en brazos o ropa con confianza de 0.55 a 0.73
        # (ver docs/analisis/unc/02_resultados_registro.md §24).
        self.confianza_minima_deteccion = 0.75
        # Por debajo de este ancho (px) no se evalúa el anti-spoofing: el modelo
        # es inestable en caras pequeñas y genera falsas alarmas de fraude.
        # Decisión del responsable, ver docs/analisis/unc/02_resultados_registro.md §20.
        self.tamano_minimo_liveness = 60
        self.votos_requeridos = 3
        self.ventana_votos = 5
        # Revalidar la identidad anclada cada 30 frames (~1 s). Cuesta ~2.4 % más de
        # cómputo y muestra "Verificando" durante 2–3 frames; no hay interfaz en el
        # producto final. Ver docs/analisis/unc/02_resultados_registro.md §27.
        self.revalidar_cada = 30
        # Dos pistas activas en el mismo frame no pueden ser la misma persona.
        # Si dos reclaman la misma identidad y la diferencia de similitud supera este
        # margen, la peor se degrada a Desconocido (ver
        # docs/analisis/unc/09_incidente_identidad_dani.md).
        self.margen_conflicto = 0.05
        # Frames que se mantiene vetada una identidad para una pista que perdió el
        # conflicto, mientras la pista ganadora siga anclada.
        self.frames_veto = 90
        self.vetos_identidad = {}
        # Por debajo de esta similitud (y por encima del umbral) la identidad se
        # muestra como "posible": el sistema no puede afirmarla con seguridad.
        # No cambia la decisión ni la asistencia, solo la presentación.
        self.umbral_revision = 0.40
```

- **Línea 117 `umbral_similitud = 0.30`:** si la mejor coincidencia no supera 0.30,
  la cara es `DESCONOCIDO`. Se eligió con la curva ROC: con 0.30 la precisión es
  0.987 y el recall 0.872; con 0.28 se gana recall pero se triplican los falsos.
  En asistencia, un falso positivo (marcar a quien no es) pesa más que un falso
  negativo (no marcarlo).
- **Línea 118 `umbral_liveness = 0.90`:** el anti-spoofing debe estar muy seguro
  para gritar "fraude". Es un umbral alto a propósito: una falsa alarma de fraude
  es peor que dejar pasar un spoof dudoso.
- **Línea 119 `tamano_minimo_rostro = 40`:** descarta caras de menos de 40 px de
  ancho. En el modo base de datos se baja a 20 porque las caras del salón son
  pequeñas; el valor 40 es el de la ventana de escritorio.
- **Línea 123 `confianza_minima_deteccion = 0.75`:** RetinaFace devuelve cajas
  falsas en brazos o ropa con confianza 0.55-0.73. El 0.75 las elimina sin perder
  caras reales.
- **Línea 127 `tamano_minimo_liveness = 60`:** por debajo de 60 px el spoofer es
  inestable y genera falsos fraudes. En esas caras se **omite** el anti-spoofing
  (no se marca como fraude).
- **Línea 128 `votos_requeridos = 3`:** tres frames consecutivos con el mismo
  candidato antes de confirmar. Es el consenso temporal que evita confirmar por un
  frame aislado.
- **Línea 129 `ventana_votos = 5`:** si pasan más de 5 frames sin votar por el
  mismo candidato, la racha se reinicia. Evita sumar votos de momentos distintos.
- **Línea 133 `revalidar_cada = 30`:** una identidad anclada se reutiliza 30 frames
  (~1 s a 30 fps) sin recalcular el embedding. Ahorra cómputo; el coste medido es
  ~2.4 %.
- **Línea 138 `margen_conflicto = 0.05`:** si dos pistas reclaman la misma
  identidad y sus similitudes difieren en 0.05 o más, la peor se degrada. Si
  difieren menos, se dejan ambas y se marca "ambigua" (puede ser un fragmento del
  tracker, no dos personas).
- **Línea 141 `frames_veto = 90`:** la pista que pierde un conflicto no puede
  reclamar esa identidad durante 90 frames (~3 s), mientras la ganadora siga
  anclada. Evita el ping-pong entre dos pistas.
- **Línea 142 `vetos_identidad = {}`:** diccionario `track_id → {nombre, contra,
  hasta}`. Se limpia cada 120 frames (línea 387).
- **Línea 146 `umbral_revision = 0.40`:** la banda 0.30-0.40 se muestra como
  "posible &lt;nombre&gt;" en amarillo. **No cambia la decisión ni la asistencia**;
  es solo presentación para que el humano revise.

### 5.3 Carga de centroides (líneas 148-171)

```python
        print("\n[INFO] Inicializando Motor de Asistencia Optimizado...")

        if usar_bd:
            # La base vectorial (PostgreSQL + pgvector) es la fuente de verdad.
            from core.db_config import cargar_centroides

            print("[INFO] Cargando centroides desde pgvector...")
            self.db = cargar_centroides()
            origen = "PostgreSQL/pgvector"
            if not self.db:
                raise ValueError(
                    "La base vectorial está vacía: registra al menos un estudiante "
                    "(python manage.py importar_centroides)."
                )
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"No se encontró la base de datos en {self.db_path}"
                )
            with open(self.db_path, "rb") as f:
                self.db = pickle.load(f)
            origen = str(self.db_path)
            if not self.db:
                raise ValueError(f"La base de datos {self.db_path} está vacía")
```

- **Línea 148:** aviso de arranque. Los modelos tardan varios segundos en cargar.
- **Línea 150:** rama principal: PostgreSQL + pgvector.
- **Línea 152:** import perezoso de `cargar_centroides`. `core/db_config.py` usa
  `psycopg2` directamente, sin Django, para que los scripts de consola funcionen.
- **Línea 155:** `self.db` es un diccionario `alias → np.ndarray(512)`.
- **Línea 156:** `origen` solo se usa para el mensaje final.
- **Línea 157-161:** si la tabla está vacía, el motor no puede hacer nada. Falla
  con un mensaje que dice cómo poblarla (`importar_centroides`).
- **Línea 162-171:** rama de respaldo con el pickle. Solo para experimentos que
  necesitan reproducir un estado histórico.
- **Línea 163-166:** si el archivo no existe, `FileNotFoundError`.
- **Línea 167-168:** carga el diccionario con `pickle.load`.
- **Línea 170-171:** un pickle vacío también es un error.

### 5.4 Validación y normalización (líneas 173-180)

```python
        self.nombres_lista = list(self.db.keys())
        vectores = np.array([np.asarray(v).reshape(-1) for v in self.db.values()], dtype="float32")
        if vectores.shape[1] != 512:
            raise ValueError(f"Dimensión de embedding inesperada: {vectores.shape[1]}")
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        if not np.allclose(normas, 1.0, atol=1e-3):
            print("[AVISO] Embeddings no normalizados en la DB; se normalizan al cargar.")
            vectores = vectores / normas
```

- **Línea 173:** `nombres_lista` fija el orden de las filas del índice. FAISS
  devuelve índices numéricos; este arreglo los traduce a nombres.
- **Línea 174:** apila los 9 vectores en una matriz `(9, 512)` de `float32`, que es
  el tipo que exige FAISS. `reshape(-1)` acepta vectores guardados como listas.
- **Línea 175-176:** si algún vector no tiene 512 dimensiones, algo está corrupto;
  mejor fallar que comparar vectores incompatibles.
- **Línea 177:** norma de cada fila: ‖v‖ = √(Σ vᵢ²). `keepdims=True` la deja como
  columna `(9, 1)` para dividir por filas.
- **Línea 178:** `np.allclose(..., atol=1e-3)` comprueba que todas las normas son
  1 con tolerancia de 0.001.
- **Línea 179-180:** si alguna no lo es, avisa y normaliza `v / ‖v‖`. Sin esto, el
  producto interno no sería una similitud coseno válida.

### 5.5 Índice FAISS (líneas 182-187)

```python
        print(
            f"[INFO] Compilando Índice FAISS para {len(self.nombres_lista)} estudiantes "
            f"desde {origen}..."
        )
        self.indice_faiss = faiss.IndexFlatIP(512)
        self.indice_faiss.add(vectores)
```

- **Línea 183-185:** mensaje con el número de estudiantes y el origen.
- **Línea 186:** `IndexFlatIP(512)` = producto interno exacto sobre 512
  dimensiones. Como los vectores están normalizados, el producto interno es
  exactamente el coseno.
- **Línea 187:** inserta las 9 filas. A partir de aquí, una búsqueda es comparar
  contra las 9, sin reconstruir nada.

### 5.6 Modelos y tracker (líneas 189-192)

```python
        self.detector = create_detector("retinaface")
        self.recognizer = create_recognizer("arcface")
        self.spoofer = create_spoofer()
        self.tracker = sv.ByteTrack()
```

- **Línea 189:** RetinaFace. Detecta cajas y 5 landmarks (ojos, nariz, comisuras).
- **Línea 190:** ArcFace. Convierte una cara alineada en un vector de 512.
- **Línea 191:** anti-spoofing. Devuelve `is_real` y `confidence`.
- **Línea 192:** ByteTrack. Asigna y mantiene `track_id` entre frames. Se
  actualiza incluso sin detecciones para "envejecer" las pistas perdidas.

### 5.7 Estado de la sesión (líneas 194-202)

```python
        self.frame_idx = 0
        self.identidades_ancladas = {}
        self.votos_identidad = {}
        self.asistencia_hoy = set()
        self.fecha_actual = datetime.now().strftime("%Y-%m-%d")

        if self.registrar_csv:
            self._preparar_archivo_csv()
        print("[OK] Sistema de Inferencia Listo.\n")
```

- **Línea 194 `frame_idx`:** contador global de frames. Se usa para la ventana de
  votos, la revalidación y los vetos.
- **Línea 195 `identidades_ancladas`:** `track_id → {nombre, similitud, frame}`.
- **Línea 196 `votos_identidad`:** `track_id → {candidato, conteo, ultimo}`.
- **Línea 197 `asistencia_hoy`:** nombres ya registrados hoy. Evita duplicados en
  el CSV y alimenta el contador de la ventana.
- **Línea 198 `fecha_actual`:** fecha de hoy en formato `YYYY-MM-DD`. Si el
  proceso cruza la medianoche, `_registrar_asistencia` reinicia el set.
- **Línea 200-201:** solo prepara el CSV si se va a escribir. En modo BD no se
  crea ningún archivo.
- **Línea 202:** mensaje de listo.

---

## 6. CSV (`_preparar_archivo_csv` y `_registrar_asistencia`, líneas 204-249)

```python
    def _preparar_archivo_csv(self):
        """Prepara el archivo CSV y carga los registros del día actual para no duplicar."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Identificador", "Fecha", "Hora", "Similitud"])
        else:
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2 and row[1] == self.fecha_actual:
                            self.asistencia_hoy.add(row[0])
            except Exception as e:
                print(f"Advertencia al leer CSV: {e}")
```

- **Línea 206:** crea la carpeta del CSV si no existe (`data/`). `parents=True`
  crea toda la ruta; `exist_ok=True` no falla si ya está.
- **Línea 208:** si el archivo no existe, lo crea y escribe solo el encabezado:
  `Identificador, Fecha, Hora, Similitud`.
- **Línea 212-221:** si ya existe, lo lee para recuperar los nombres de hoy.
- **Línea 215-216:** `next(reader, None)` salta el encabezado sin fallar si el
  archivo está vacío.
- **Línea 218:** solo cuenta las filas cuya segunda columna (fecha) es hoy. Así,
  reiniciar el motor el mismo día no duplica la asistencia.
- **Línea 220-221:** si el CSV está corrupto, avisa y sigue; no bloquea el motor.

```python
    def _registrar_asistencia(self, nombre, similitud):
        """Escribe en el CSV. Reinicia el conjunto diario si cambió el día.

        En modo base de datos (`registrar_csv=False`) no hace nada: la escritura
        la resuelve `core.registro_asistencia.aplicar_decisiones`.
        """
        if not self.registrar_csv:
            return

        ahora = datetime.now()
        fecha = ahora.strftime("%Y-%m-%d")
        if fecha != self.fecha_actual:
            self.fecha_actual = fecha
            self.asistencia_hoy = set()

        if nombre in self.asistencia_hoy or nombre == "Desconocido":
            return

        hora = ahora.strftime("%H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([nombre, fecha, hora, f"{similitud:.4f}"])

        self.asistencia_hoy.add(nombre)
        print(
            f"  ASISTENCIA CONFIRMADA: {nombre} | Similitud: {similitud:.2f} | Hora: {hora}"
        )
```

- **Línea 229-230:** el interruptor. En modo BD, la función termina aquí; la
  escritura la hace `aplicar_decisiones`.
- **Línea 232-233:** fecha y hora actuales.
- **Línea 234-236:** si cambió el día, reinicia `asistencia_hoy` (el proceso pudo
  quedar corriendo de un día para otro).
- **Línea 238-239:** no duplica un nombre ya registrado hoy y nunca registra
  "Desconocido".
- **Línea 241:** hora en formato `HH:MM:SS`.
- **Línea 242-244:** abre en modo append (`"a"`) y escribe cuatro columnas. La
  similitud se formatea con 4 decimales para que el CSV sea estable.
- **Línea 246:** marca el nombre como registrado en memoria.
- **Línea 247-249:** imprime la confirmación en la consola.

---

## 7. `_calcular_iou()` (líneas 251-260)

```python
    @staticmethod
    def _calcular_iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
```

Calcula el **IoU** (Intersection over Union): cuánto se solapan dos cajas.

**Fórmula:**

```
IoU = área(A ∩ B) / área(A ∪ B)
    = área(A ∩ B) / (área(A) + área(B) − área(A ∩ B))
```

- **Línea 253:** esquina superior izquierda de la intersección: el máximo de las
  coordenadas izquierdas y de las superiores.
- **Línea 254:** esquina inferior derecha: el mínimo de las derechas e inferiores.
- **Línea 255:** área de la intersección. Si las cajas no se tocan, `xB - xA` o
  `yB - yA` es negativo; `max(0, ...)` lo convierte en 0.
- **Línea 256-257:** sin intersección, IoU = 0.
- **Línea 258-259:** áreas individuales.
- **Línea 260:** divide la intersección por la unión. El `float(...)` evita
  división entera en casos extremos.

**Dónde se usa:** `procesar_frame` empareja cada caja del tracker con una
detección de RetinaFace. Si el mejor IoU es menor que 0.5, la pista no se evalúa
(línea 431). Sin ese emparejamiento no se sabría a qué landmarks corresponde cada
`track_id`.

---

## 8. `_contar_voto()` (líneas 262-273)

```python
    def _contar_voto(self, t_id, candidato):
        voto = self.votos_identidad.get(t_id)
        if (
            voto is None
            or voto["candidato"] != candidato
            or self.frame_idx - voto["ultimo"] > self.ventana_votos
        ):
            voto = {"candidato": candidato, "conteo": 0, "ultimo": self.frame_idx}
            self.votos_identidad[t_id] = voto
        voto["conteo"] += 1
        voto["ultimo"] = self.frame_idx
        return voto["conteo"]
```

- **Línea 263:** recupera el voto acumulado de esa pista, si existe.
- **Línea 264-268:** reinicia la racha en tres casos:
  1. no había voto (`None`),
  2. el candidato cambió (un frame dijo `dani`, el siguiente `alexa`),
  3. pasaron más de 5 frames desde el último voto.
- **Línea 269:** crea un voto nuevo con conteo 0 en el frame actual.
- **Línea 271:** suma el voto de este frame.
- **Línea 272:** actualiza el frame del último voto.
- **Línea 273:** devuelve el conteo actualizado.

El motor confirma cuando este conteo llega a `votos_requeridos` (3). Es un
consenso de **3 frames consecutivos** (con tolerancia de 5 frames de ventana).

---

## 9. `_evaluar_liveness()` (líneas 275-285)

```python
    def _evaluar_liveness(self, frame, bbox):
        ancho = bbox[2] - bbox[0]
        if ancho < self.tamano_minimo_liveness:
            return "omitido"
        try:
            res = self.spoofer.predict(frame, bbox)
        except Exception:
            return "error"
        if res.is_real and res.confidence >= self.umbral_liveness:
            return "real"
        return "fraude"
```

- **Línea 276:** ancho de la cara en píxeles.
- **Línea 277-278:** si mide menos de 60 px, devuelve `"omitido"`. No es fraude ni
  real: simplemente no se evalúa porque el modelo es inestable en caras pequeñas.
- **Línea 279-282:** si el spoofer lanza una excepción, devuelve `"error"`. El
  error del modelo **no** se trata como fraude.
- **Línea 283-284:** solo es `"real"` si el modelo lo dice y además tiene
  confianza ≥ 0.90.
- **Línea 285:** cualquier otro caso es `"fraude"`.

Valores posibles: `"omitido"`, `"real"`, `"fraude"`, `"error"`.

---

## 10. `_decidir_identidad()` (líneas 287-336) — el corazón

```python
    def _decidir_identidad(self, frame, t_id, rostro):
        ancla = self.identidades_ancladas.get(t_id)
        if ancla is not None:
            if self.frame_idx - ancla["frame"] < self.revalidar_cada:
                return ancla["nombre"], ancla["similitud"], "ANCLADO", None, None, ancla["nombre"]
            del self.identidades_ancladas[t_id]

        vec = embedding_facial(self.recognizer, frame, rostro.landmarks)
        dist, idx = self.indice_faiss.search(vec.reshape(1, -1).astype("float32"), 2)
        sim = float(dist[0][0])
        sim2 = float(dist[0][1])
        nombre2 = self.nombres_lista[idx[0][1]]

        if sim <= self.umbral_similitud:
            self.votos_identidad.pop(t_id, None)
            return "Desconocido", sim, "DESCONOCIDO", sim2, nombre2, self.nombres_lista[idx[0][0]]

        candidato = self.nombres_lista[idx[0][0]]

        # Si esta pista perdió un conflicto de identidad, se mantiene como
        # Desconocido mientras la pista ganadora siga anclada.
        veto = self.vetos_identidad.get(t_id)
        if (
            veto
            and veto["nombre"] == candidato
            and self.frame_idx <= veto["hasta"]
            and veto["contra"] in self.identidades_ancladas
        ):
            self.votos_identidad.pop(t_id, None)
            return "Desconocido", sim, "DESCONOCIDO", sim2, nombre2, candidato

        liveness = self._evaluar_liveness(frame, rostro.bbox)
        if liveness == "fraude":  # "omitido" (cara pequeña) continúa como válido
            self.votos_identidad.pop(t_id, None)
            return "FRAUDE DETECTADO", sim, "FRAUDE", sim2, nombre2, candidato
        if liveness == "error":
            return "Error liveness", sim, "ERROR", sim2, nombre2, candidato

        conteo = self._contar_voto(t_id, candidato)
        if conteo >= self.votos_requeridos:
            self.identidades_ancladas[t_id] = {
                "nombre": candidato,
                "similitud": sim,
                "frame": self.frame_idx,
            }
            self.votos_identidad.pop(t_id, None)
            self._registrar_asistencia(candidato, sim)
            return candidato, sim, "CONFIRMADO", sim2, nombre2, candidato

        return f"Verificando {conteo}/{self.votos_requeridos}", sim, "VERIFICANDO", sim2, nombre2, candidato
```

Esta función se llama una vez por pista y por frame. Devuelve **seis valores**:

```
(nombre, similitud, estado, sim_top2, nombre2, candidato)
```

### 10.1 Ancla (líneas 288-292)

- **Línea 288:** busca si la pista ya tiene una identidad confirmada
  (`identidades_ancladas`).
- **Línea 290:** si el ancla tiene menos de 30 frames, se reutiliza.
- **Línea 291:** devuelve el nombre y la similitud guardados, estado `ANCLADO`, y
  `None` para top-2 (no se recalculó). El sexto valor es el propio nombre como
  candidato.
- **Línea 292:** si el ancla tiene 30 frames o más, se borra y se recalcula.

**Por qué existe el ancla:** recalcular el embedding en cada frame es caro y, en
una cara que ya está confirmada, innecesario. Se revalida cada segundo para
detectar si la persona cambió o si otra ocupó su lugar.

### 10.2 Búsqueda en FAISS (líneas 294-298)

- **Línea 294:** `embedding_facial` alinea la cara con los landmarks, aplica el
  preprocesado (CLAHE, etc.) y devuelve el vector de 512. Es **el mismo** que usa
  el registro.
- **Línea 295:** `search(vec, 2)` pide los **dos** centroides más parecidos.
  `dist` son las similitudes; `idx`, sus posiciones.
- **Línea 296:** `sim` = mejor similitud (top-1).
- **Línea 297:** `sim2` = segunda mejor (top-2).
- **Línea 298:** `nombre2` = nombre del segundo. Se guarda para análisis; **no
  decide**.

### 10.3 Umbral (líneas 300-302)

- **Línea 300:** si la mejor similitud es ≤ 0.30, la cara es desconocida.
- **Línea 301:** borra los votos de esa pista. Un frame desconocido rompe la racha.
- **Línea 302:** devuelve `"Desconocido"`, estado `DESCONOCIDO`, y como candidato
  el top-1 (aunque no haya pasado el umbral). Ese candidato es el que luego se
  escribe en la nota del intruso: "mejor coincidencia: dani (0.14)".

> **Detalle:** el umbral usa `<=`, no `<`. Una similitud exactamente 0.30 se
> considera desconocida. Es coherente con la tabla ROC, donde 0.30 es el punto de
> operación.

### 10.4 Veto por conflicto (líneas 304-316)

- **Línea 304:** `candidato` es el top-1 de este frame.
- **Línea 308:** busca un veto para esta pista.
- **Línea 309-314:** el veto aplica solo si:
  1. la pista tiene veto,
  2. el nombre vetado coincide con el candidato actual,
  3. todavía no expiró (`frame_idx <= hasta`),
  4. la pista ganadora **sigue anclada**.
- **Línea 315-316:** si aplica, devuelve `DESCONOCIDO` y borra los votos. La pista
  no puede reclamar una identidad que perdió mientras la ganadora siga presente.

### 10.5 Anti-spoofing (líneas 318-323)

- **Línea 318:** evalúa liveness.
- **Línea 319-321:** si es fraude, borra votos y devuelve `FRAUDE DETECTADO`
  (estado `FRAUDE`).
- **Línea 322-323:** si hubo error del modelo, devuelve `ERROR` **sin borrar
  votos**: el error no es culpa de la cara.
- `"omitido"` (cara pequeña) **no** entra en ninguno de los dos `if`: sigue el
  flujo normal como si fuera real.

### 10.6 Consenso y confirmación (líneas 325-336)

- **Línea 325:** suma un voto para el candidato.
- **Línea 326:** si llegó a 3 votos…
- **Línea 327-331:** guarda el ancla con nombre, similitud y frame actual.
- **Línea 332:** limpia los votos (ya no hacen falta).
- **Línea 333:** escribe la asistencia. En modo CSV escribe la fila; en modo BD no
  hace nada (lo resuelve la capa web).
- **Línea 334:** devuelve estado `CONFIRMADO`.
- **Línea 336:** si aún no hay 3 votos, devuelve `VERIFICANDO` con el conteo
  (`Verificando 2/3`). Ese texto se muestra en la ventana.

---

## 11. `_marcar_revision()` (líneas 338-351)

```python
    def _marcar_revision(self, decisiones):
        """Marca las identidades de baja confianza para revisión visual.

        No cambia la decisión ni la asistencia: solo indica que la similitud está
        entre el umbral y `umbral_revision`, donde el sistema no puede afirmar la
        identidad con seguridad (banda 0.30–0.40).
        """
        for decision in decisiones:
            confirmada = decision["estado"] in ("CONFIRMADO", "ANCLADO")
            sim = decision.get("similitud")
            decision["revision"] = bool(
                confirmada and sim is not None and sim < self.umbral_revision
            )
        return decisiones
```

- **Línea 345:** recorre todas las decisiones del frame.
- **Línea 346:** solo las identidades confirmadas o ancladas pueden estar "en
  revisión".
- **Línea 347:** recupera la similitud (puede ser `None` en `CENTRATE`).
- **Línea 348-350:** `revision` es `True` solo si está confirmada, tiene
  similitud y es menor que 0.40. Es decir, la banda **0.30-0.40**.
- **Línea 351:** devuelve la misma lista, ya modificada.

**Qué hace con eso la UI:** `servicio_en_vivo._dibujar` pinta la caja de amarillo y
escribe `posible <nombre>`. La ventana de escritorio no lo usa: muestra el nombre
normal. **La asistencia se registra igual**: la banda no cambia la decisión.

---

## 12. `_resolver_conflictos()` (líneas 353-392)

```python
    def _resolver_conflictos(self, decisiones):
        """Impide que dos pistas simultáneas compartan la misma identidad.

        Dos caras no pueden ser la misma persona en el mismo frame. Si dos pistas
        confirman el mismo nombre, se conserva la de mayor similitud y la otra vuelve
        a Desconocido. Si la diferencia es menor que `margen_conflicto`, se dejan ambas
        y se marca el motivo como ambiguo (puede ser un fragmento del tracker).
        """
        por_nombre = {}
        for decision in decisiones:
            if decision["estado"] in ("CONFIRMADO", "ANCLADO") and decision.get("similitud") is not None:
                por_nombre.setdefault(decision["nombre"], []).append(decision)

        for nombre, grupo in por_nombre.items():
            if len(grupo) < 2:
                continue
            grupo.sort(key=lambda d: d["similitud"], reverse=True)
            ganadora = grupo[0]
            for otra in grupo[1:]:
                diferencia = ganadora["similitud"] - otra["similitud"]
                if diferencia >= self.margen_conflicto:
                    self.identidades_ancladas.pop(otra["track_id"], None)
                    self.votos_identidad.pop(otra["track_id"], None)
                    self.vetos_identidad[otra["track_id"]] = {
                        "nombre": nombre,
                        "contra": ganadora["track_id"],
                        "hasta": self.frame_idx + self.frames_veto,
                    }
                    otra["estado"] = "DESCONOCIDO"
                    otra["nombre"] = "Desconocido"
                    otra["motivo"] = f"conflicto_con_pista_{ganadora['track_id']}"
                else:
                    otra["motivo"] = f"identidad_ambigua_con_pista_{ganadora['track_id']}"

        if self.frame_idx % 120 == 0:
            self.vetos_identidad = {
                track: veto for track, veto in self.vetos_identidad.items()
                if veto["hasta"] > self.frame_idx
            }
        return decisiones
```

Esta regla nació del **incidente de identidad** (2026-10-21): dos caras aparecieron
etiquetadas como `dani` al mismo tiempo. La causa no era el umbral, sino que dos
pistas distintas reclamaban la misma identidad.

- **Línea 361:** diccionario `nombre → lista de decisiones`.
- **Línea 362-364:** agrupa solo las decisiones confirmadas o ancladas con
  similitud. Las demás no pueden estar en conflicto.
- **Línea 366:** recorre cada nombre.
- **Línea 367-368:** si solo hay una pista con ese nombre, no hay conflicto.
- **Línea 369:** ordena el grupo por similitud descendente. La primera es la
  ganadora.
- **Línea 371:** recorre las demás pistas del grupo.
- **Línea 372:** diferencia de similitud entre la ganadora y esta.
- **Línea 373:** si la diferencia es ≥ 0.05, hay un ganador claro.
- **Línea 374-375:** desancla la pista perdedora y borra sus votos.
- **Línea 376-380:** le pone un veto de 90 frames: no puede reclamar ese nombre
  mientras la ganadora siga anclada.
- **Línea 381-383:** la decisión pasa a `DESCONOCIDO` y se anota el motivo:
  `conflicto_con_pista_<id>`. Ese motivo es el que `SeguimientoIntrusos` usa para
  **no** registrar un intruso falso.
- **Línea 384-385:** si la diferencia es menor que 0.05, se dejan ambas, pero se
  marca `identidad_ambigua_con_pista_<id>`. Puede ser el mismo rostro partido en
  dos pistas por el tracker.
- **Línea 387-391:** cada 120 frames limpia los vetos expirados. Evita que el
  diccionario crezca sin límite en un video largo.

**Por qué el margen es 0.05:** con dos caras reales, la similitud de la correcta
suele superar a la impostora por más de 0.05. Con dos fragmentos de la misma cara,
la diferencia es menor. El margen separa "conflicto real" de "fragmento".

---

## 13. `procesar_frame()` (líneas 394-446)

```python
    def procesar_frame(self, frame):
        """Procesa un frame y devuelve decisiones estructuradas (sin dibujar ni mostrar)."""
        self.frame_idx += 1
        h_frame, w_frame = frame.shape[:2]
        margen = 15

        rostros = [
            r for r in self.detector.detect(frame)
            if (r.bbox[2] - r.bbox[0]) > self.tamano_minimo_rostro
            and float(r.confidence) >= self.confianza_minima_deteccion
        ]

        if rostros:
            detections = sv.Detections(
                xyxy=np.array([r.bbox for r in rostros], dtype=float),
                confidence=np.array([r.confidence for r in rostros], dtype=float),
            )
        else:
            detections = sv.Detections.empty()
        tracked = self.tracker.update_with_detections(detections=detections)

        decisiones = []
        for t in tracked:
            x1, y1, x2, y2 = map(int, t[0])
            t_id = t[4]

            if x1 < margen or y1 < margen or x2 > (w_frame - margen) or y2 > (h_frame - margen):
                decisiones.append({"track_id": t_id, "bbox": (x1, y1, x2, y2),
                                   "nombre": "Céntrate", "similitud": None, "estado": "CENTRATE"})
                continue

            mejor_iou, r_asociado = 0, None
            for r in rostros:
                iou = self._calcular_iou([x1, y1, x2, y2], list(map(int, r.bbox)))
                if iou > mejor_iou:
                    mejor_iou, r_asociado = iou, r

            if mejor_iou < 0.5 or r_asociado is None:
                continue

            nombre, sim, estado, sim2, nombre2, candidato = self._decidir_identidad(frame, t_id, r_asociado)
            decisiones.append({
                "track_id": t_id, "bbox": (x1, y1, x2, y2),
                "nombre": nombre, "similitud": sim, "estado": estado,
                "confianza_det": round(float(r_asociado.confidence), 4),
                "sim_top2": None if sim2 is None else round(sim2, 4),
                "nombre2": nombre2,
                "candidato": candidato,
                "margen": None if sim2 is None else round(sim - sim2, 4),
            })

        self._marcar_revision(decisiones)
        return self._resolver_conflictos(decisiones)
```

- **Línea 396:** incrementa el contador global. Es la referencia temporal de todo
  el estado (votos, anclas, vetos).
- **Línea 397:** alto y ancho del frame, para el margen de borde.
- **Línea 398:** margen de 15 px. Las caras pegadas al borde se marcan
  `CENTRATE`: están cortadas y el embedding sería malo.
- **Línea 400-404:** detección con RetinaFace y dos filtros:
  - ancho > `tamano_minimo_rostro` (40 px, o 20 en modo BD),
  - confianza ≥ 0.75.
- **Línea 406-412:** convierte las detecciones al formato de `supervision`. Si no
  hay ninguna, usa `Detections.empty()` para que ByteTrack pueda envejecer las
  pistas perdidas.
- **Línea 413:** actualiza el tracker y devuelve las pistas activas. Cada fila
  trae `[x1, y1, x2, y2, track_id, ...]`.
- **Línea 415:** lista de decisiones del frame.
- **Línea 417-418:** extrae la caja y el `track_id`.
- **Línea 420-423:** si la caja toca el borde, añade `CENTRATE` y salta la
  decisión. No se gasta cómputo en una cara incompleta.
- **Línea 425-429:** empareja la pista con la detección de mayor IoU.
- **Línea 431-432:** si el mejor IoU es < 0.5, no hay correspondencia fiable; se
  omite la pista en este frame.
- **Línea 434:** llama al corazón de la decisión.
- **Línea 435-443:** construye el diccionario de decisión con:
  - `track_id`, `bbox`, `nombre`, `similitud`, `estado`,
  - `confianza_det`: confianza del detector, redondeada a 4 decimales,
  - `sim_top2`: segunda similitud o `None`,
  - `nombre2`: segundo nombre o `None`,
  - `candidato`: top-1 aunque no haya pasado el umbral,
  - `margen`: `sim - sim2` o `None`.
- **Línea 445:** marca la banda de revisión.
- **Línea 446:** resuelve conflictos y devuelve la lista final.

**Campos de una decisión:**

| Campo | Tipo | Significado |
|---|---|---|
| `track_id` | int | Pista de ByteTrack |
| `bbox` | (x1, y1, x2, y2) | Caja en píxeles |
| `nombre` | str | Identidad, `Desconocido`, `Céntrate`... |
| `similitud` | float o None | Similitud top-1 |
| `estado` | str | `CONFIRMADO`, `ANCLADO`, `VERIFICANDO`, `DESCONOCIDO`, `FRAUDE`, `ERROR`, `CENTRATE` |
| `confianza_det` | float | Confianza del detector |
| `sim_top2` | float o None | Segunda similitud |
| `nombre2` | str o None | Segundo candidato |
| `candidato` | str o None | Top-1 aunque no pase el umbral |
| `margen` | float o None | `sim - sim2` |
| `motivo` | str o None | `conflicto_...` o `identidad_ambigua_...` |
| `revision` | bool | Banda 0.30-0.40 |

---

## 14. `iniciar_vigilancia()` (líneas 448-521)

```python
    def iniciar_vigilancia(self, origen_video=0, sesion_id=None, limpiar=False):
        """Procesa una cámara o un video.

        Con `sesion_id` escribe la asistencia en la base (RegistroAsistencia) con
        las mismas reglas que la web y deja el CSV desactivado. Sin él se comporta
        como siempre: dibuja en pantalla y, si `registrar_csv` está activo,
        escribe el CSV de los experimentos. Con `limpiar` borra la sesión de demo
        al terminar (solo tiene sentido con `--crear-sesion`).
        """
        sesion = estudiantes = profesor = alias_profesor = seguimiento = None
        if sesion_id is not None:
            self.registrar_csv = False
            # Mismo filtro de tamaño que los flujos web (batch y en vivo).
            self.tamano_minimo_rostro = 20
            sesion, estudiantes, profesor, alias_profesor = _contexto_sesion(sesion_id)
            from core.intrusos import SeguimientoIntrusos
            from core.registro_asistencia import aplicar_decisiones

            seguimiento = SeguimientoIntrusos()

        cap = cv2.VideoCapture(origen_video)
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara o flujo de video.")
            return

        cv2.namedWindow("Control de Asistencia", cv2.WINDOW_NORMAL)
        tiempo_previo = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                decisiones = self.procesar_frame(frame)

                if sesion is not None:
                    resumen = aplicar_decisiones(
                        decisiones, frame, sesion, estudiantes,
                        alias_profesor=alias_profesor, profesor=profesor,
                        seguimiento=seguimiento,
                    )
                    self.asistencia_hoy.update(resumen["confirmados"])

                for d in decisiones:
                    x1, y1, x2, y2 = d["bbox"]
                    self._dibujar_ui(frame, x1, y1, x2, y2, d["nombre"],
                                     COLORES.get(d["estado"], (0, 0, 255)), d["track_id"])

                t_actual = time.time()
                fps = 1 / (t_actual - tiempo_previo) if (t_actual - tiempo_previo) > 0 else 0
                tiempo_previo = t_actual

                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Asistencia Hoy: {len(self.asistencia_hoy)}",
                            (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Control de Asistencia", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if limpiar and sesion is not None:
                from core.sesion_demo import limpiar_sesion_demo

                limpiar_sesion_demo(sesion)
                print("\nSesión de demo eliminada.")
            print(f"\n{'=' * 50}")
            print(" SESIÓN DE ASISTENCIA CERRADA")
            print(f" Estudiantes procesados hoy: {len(self.asistencia_hoy)}")
            print(f"{'=' * 50}")
```

- **Línea 457:** inicializa a `None` todas las variables de contexto. Así el bloque
  `finally` puede consultarlas aunque no se haya entrado en modo BD.
- **Línea 458:** si hay `sesion_id`, es modo base de datos.
- **Línea 459:** desactiva el CSV. La base es la única fuente de verdad.
- **Línea 461:** baja el filtro de cara a 20 px, igual que la web. El salón tiene
  caras pequeñas.
- **Línea 462:** carga sesión, inscritos, profesor y alias.
- **Línea 463-464:** imports perezosos de la capa web.
- **Línea 466:** crea el clasificador de intrusos, que decide cuándo un
  `DESCONOCIDO` es intruso real (mejor score < 0.18, 15 frames, sin conflictos).
- **Línea 468:** abre la fuente. `0` es la cámara; un string, un archivo.
- **Línea 469-471:** si no abre, avisa y sale.
- **Línea 473:** crea la ventana redimensionable.
- **Línea 474:** tiempo del frame anterior para el cálculo de FPS.
- **Línea 477:** bucle infinito hasta que se acabe el video o se pulse `q`.
- **Línea 478-480:** lee un frame; si falla (fin del video), rompe.
- **Línea 482:** procesa el frame y obtiene las decisiones.
- **Línea 484-490:** en modo BD, escribe la asistencia y actualiza el contador de
  la ventana con los confirmados del frame.
- **Línea 492-495:** dibuja cada decisión con su color.
- **Línea 497-498:** calcula los FPS instantáneos:
  `fps = 1 / (t_actual − t_previo)`. Protege la división por cero.
- **Línea 499:** actualiza el tiempo previo.
- **Línea 501-504:** dibuja FPS arriba y "Asistencia Hoy" abajo.
- **Línea 506-508:** muestra el frame y escucha la tecla `q`. `waitKey(1)` espera
  1 ms; sin él, la ventana no se refresca.
- **Línea 510-521:** `finally` garantiza la limpieza aunque haya una excepción:
  - libera la cámara,
  - cierra las ventanas,
  - si `limpiar`, borra la sesión de demo,
  - imprime el resumen final.

> **Nota:** en modo BD, `aplicar_decisiones` puede **borrar** filas de intruso si
> la pista luego mejora. Por eso el contador de la ventana puede subir y bajar
> durante la captura.

---

## 15. `_dibujar_ui()` (líneas 523-536)

```python
    def _dibujar_ui(self, frame, x1, y1, x2, y2, nombre, color, t_id):
        """Dibuja las cajas y etiquetas de forma limpia en el frame."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        etiqueta = nombre if nombre == "FRAUDE DETECTADO" else f"{nombre} (ID:{t_id})"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        color_texto = (
            (0, 0, 0)
            if color == (0, 255, 0) or color == (0, 165, 255)
            else (255, 255, 255)
        )
        cv2.putText(
            frame, etiqueta, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2
        )
```

- **Línea 525:** dibuja el rectángulo de la cara con grosor 2.
- **Línea 526:** la etiqueta. Para `FRAUDE DETECTADO` no añade `(ID:n)` porque el
  texto sería muy largo; para el resto, sí.
- **Línea 527:** mide el texto para dimensionar el fondo.
- **Línea 528:** dibuja un rectángulo relleno (`-1`) encima de la caja, del mismo
  color, para que el texto se lea.
- **Línea 529-533:** color del texto: negro sobre verde/naranja (fondos claros),
  blanco sobre rojo.
- **Línea 534-536:** escribe la etiqueta 5 px encima del borde superior.

---

## 16. `__main__` (líneas 542-595)

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Motor de asistencia sobre cámara o archivo de video."
    )
    parser.add_argument(
        "--fuente", default="0",
        help="Cámara (0, 1, ...) o ruta de un archivo de video.",
    )
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--sesion", type=int,
        help="ID de la SesionClase donde escribir la asistencia (base de datos). "
             "Sin esta opción solo se dibuja y se escribe el CSV.",
    )
    grupo.add_argument(
        "--crear-sesion", action="store_true",
        help="Crea (o reutiliza) una sesión de demo con los estudiantes "
             "y escribe la asistencia ahí.",
    )
    parser.add_argument(
        "--limpiar", action="store_true",
        help="Borra la sesión de demo al terminar (requiere --crear-sesion).",
    )
    args = parser.parse_args()

    if args.limpiar and not args.crear_sesion:
        parser.error("--limpiar requiere --crear-sesion")

    origen = int(args.fuente) if args.fuente.isdigit() else args.fuente

    try:
        sesion_id = args.sesion
        if args.crear_sesion:
            from core.sesion_demo import crear_sesion_demo

            sesion = crear_sesion_demo()
            sesion_id = sesion.id
            print(
                f"[INFO] Sesión de demo: id={sesion_id} "
                f"({sesion.asignacion_clase.asignatura.nombre})"
            )

        sistema = MotorAsistencia(registrar_csv=sesion_id is None)
        sistema.iniciar_vigilancia(
            origen_video=origen, sesion_id=sesion_id, limpiar=args.limpiar
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        if sesion_id is None:
            print(
                "Asegúrate de ejecutar el módulo de Registro y crear la base de datos primero."
            )
```

- **Línea 542:** solo se ejecuta si el archivo se lanza directamente
  (`python scripts/...`). Si se importa desde Django, no hace nada.
- **Línea 543:** `argparse` se importa aquí para no cargarlo cuando el módulo se
  usa como librería.
- **Línea 545-547:** el parser y su descripción.
- **Línea 548-551:** `--fuente`. Por defecto `"0"` (cámara). Puede ser `"1"`,
  `"2"` o una ruta de video.
- **Línea 552:** grupo mutuamente excluyente. No se puede pasar `--sesion` y
  `--crear-sesion` a la vez.
- **Línea 553-557:** `--sesion ID`: escribe en una sesión existente.
- **Línea 558-562:** `--crear-sesion`: crea una sesión de demo nueva con los
  estudiantes y escribe ahí. El texto dice "crea (o reutiliza)" pero el
  comportamiento actual **siempre crea una nueva** (ver `core/sesion_demo.py`).
- **Línea 563-566:** `--limpiar`: borra la sesión de demo al terminar.
- **Línea 567:** parsea los argumentos.
- **Línea 569-570:** validación: `--limpiar` solo tiene sentido si se creó la
  sesión. Si no, error de argparse.
- **Línea 572:** convierte la fuente: si es numérica, a `int` (índice de cámara);
  si no, se deja como ruta.
- **Línea 574:** bloque protegido para errores esperables.
- **Línea 575:** `sesion_id` inicial. Si no hay sesión, queda `None`.
- **Línea 576-584:** si se pidió crear sesión, la crea y anuncia su id y
  asignatura.
- **Línea 586:** construye el motor. `registrar_csv` es `False` si hay sesión
  (modo BD); `True` si no (modo experimento).
- **Línea 587-589:** arranca la vigilancia con la fuente, la sesión y el flag de
  limpieza.
- **Línea 590-595:** captura `FileNotFoundError` (pickle ausente) y `ValueError`
  (base vacía o sesión inexistente). Si no hay sesión, añade la pista de cómo
  poblarla.

**Ejemplos de uso:**

```bash
# Ventana con cámara, sin escribir en la base (solo CSV)
python scripts/asistencia_unificada_optimizada.py

# Ventana con un video, sin base
python scripts/asistencia_unificada_optimizada.py --fuente data/unc/salon/VID_20260907_102851.mp4

# Escribir en una sesión existente
python scripts/asistencia_unificada_optimizada.py --sesion 12 --fuente 0

# Crear sesión de demo y escribir ahí
python scripts/asistencia_unificada_optimizada.py --crear-sesion --fuente data/unc/salon/VID_20260907_102851.mp4

# Crear sesión de demo y borrarla al terminar
python scripts/asistencia_unificada_optimizada.py --crear-sesion --limpiar --fuente 0
```

---

## 17. Secuencia completa de una decisión

Para una pista concreta en un frame:

```
1. procesar_frame()
   ├─ detector.detect(frame)            → cajas + landmarks
   ├─ filtra ancho > 40 px y conf ≥ 0.75
   ├─ tracker.update_with_detections()  → track_id
   ├─ si toca el borde                 → CENTRATE
   ├─ empareja por IoU ≥ 0.5          → rostro asociado
   └─ _decidir_identidad()
       ├─ ¿ancla < 30 frames?          → ANCLADO
       ├─ embedding_facial()           → vector 512
       ├─ FAISS.search(k=2)            → sim, sim2, candidato, nombre2
       ├─ sim ≤ 0.30                   → DESCONOCIDO
       ├─ ¿veto vigente?               → DESCONOCIDO
       ├─ liveness                     → FRAUDE / ERROR / sigue
       ├─ _contar_voto()               → conteo
       ├─ conteo ≥ 3                   → CONFIRMADO + ancla + registro
       └─ si no                        → VERIFICANDO n/3
2. _marcar_revision()                   → banda 0.30-0.40
3. _resolver_conflictos()               → degrada duplicados y crea vetos
```

---

## 18. Parámetros y justificación

| Parámetro | Valor | Por qué |
|---|---|---|
| `umbral_similitud` | 0.30 | ROC: P 0.987, R 0.872; 1 FP en 157 desconocidos |
| `umbral_liveness` | 0.90 | Evitar falsas alarmas de fraude |
| `tamano_minimo_rostro` | 40 px (20 en BD) | Caras menores no son fiables |
| `confianza_minima_deteccion` | 0.75 | RetinaFace inventa caras en brazos con 0.55-0.73 |
| `tamano_minimo_liveness` | 60 px | El spoofer es inestable por debajo |
| `votos_requeridos` | 3 | Consenso temporal |
| `ventana_votos` | 5 frames | No sumar votos de momentos distintos |
| `revalidar_cada` | 30 frames | ~1 s; ahorra ~2.4 % de cómputo |
| `margen_conflicto` | 0.05 | Separa conflicto real de fragmento del tracker |
| `frames_veto` | 90 frames | ~3 s de veto |
| `umbral_revision` | 0.40 | Banda de presentación, no de decisión |

---

## 19. Complejidad y rendimiento

- **Detección:** RetinaFace es lo más caro del frame: **47 ms** por frame 1080p
  (~35 % del tiempo total medido).
- **Embedding:** ArcFace + CLAHE cuesta **24 ms por cara** (~62 %). Se calcula solo
  si no hay ancla (< 30 frames). Con ancla, el frame es casi gratis.
- **Búsqueda:** FAISS `IndexFlatIP` con 9 vectores es O(9·512): **0.02 ms por cara**,
  despreciable.
- **Liveness:** se omite en caras < 60 px y solo se evalúa cuando hay candidato
  por encima del umbral.
- **Ritmo real:** ~**7 fps** en video 1080p sobre CPU (medido entre 6.6 y 7.5 según
  el tramo). El detalle del hardware y del desglose está en la sección 21.

---

## 20. Lo que este archivo NO hace

- **No decide si un desconocido es intruso**: eso lo hace `core/intrusos.py`.
- **No escribe en PostgreSQL**: eso lo hace `core/registro_asistencia.py`.
- **No crea sesiones**: eso lo hace `core/sesion_demo.py` o el panel del profesor.
- **No dibuja la UI web**: `core/servicio_en_vivo.py` tiene su propio `_dibujar`.
- **No usa el margen top-2 para decidir**: se guarda para análisis.

---

## 21. Entorno de cómputo

Todas las mediciones se hicieron en **CPU** (Intel i9-14900KF, 62 GB de RAM), sin
usar la RTX 4090 disponible: `requirements.txt` instala `onnxruntime` (build de
CPU), no `onnxruntime-gpu`. Rendimiento medido: **~7 fps** en 1080p (detección
47 ms/frame, embedding 24 ms/cara, FAISS 0.02 ms/cara).

El detalle completo (hardware, versiones, modelos, tiempos por video, cómo
reproducirlo y cómo migrar a GPU) está en **`docs/computo.md`**.

---

## 22. Referencias cruzadas

| Tema | Documento |
|---|---|
| Curva ROC y umbral 0.30 | `docs/analisis/unc/04_resultados_curva_roc.md` |
| Registro de los 9 | `docs/analisis/unc/02_resultados_registro.md` |
| Accesorios | `docs/analisis/unc/03_resultados_robustez_accesorios.md` |
| Incidente de identidad | `docs/analisis/unc/09_incidente_identidad_dani.md` |
| Intrusos | `docs/analisis/unc/11_intrusos.md` |
| Base vectorial | `docs/base_datos.md` |
| Comandos | `docs/comandos.md` |

---

*Fin del capítulo 1. El capítulo 2 (`docs/motor_2.md`) cubre `core/utils_facial.py`,
`core/intrusos.py`, `core/db_config.py` y `core/registro_asistencia.py`.*
