# Motor de asistencia — Documentación técnica

## Capítulo 2 — Preprocesado, intrusos, base vectorial y escritura

**Archivos cubiertos:**

| Archivo | Líneas | Rol |
|---|---|---|
| `core/utils_facial.py` | 72 | Alineación, CLAHE, nitidez y embedding compartido |
| `core/intrusos.py` | 120 | Decide cuándo un desconocido es intruso real |
| `core/db_config.py` | 132 | Lectura/escritura de centroides en pgvector |
| `core/registro_asistencia.py` | 344 | Traduce decisiones a filas de `RegistroAsistencia` |

**Cómo encajan:**

```
MotorAsistencia.procesar_frame()
   │
   ├─ core.utils_facial.embedding_facial()      vector de 512
   ├─ core.db_config.cargar_centroides()        centroides de pgvector
   │
   └─ decisiones
        ├─ core.intrusos.SeguimientoIntrusos    ¿es intruso?
        └─ core.registro_asistencia
             ├─ aplicar_decisiones()            flujo en vivo
             └─ acumular + registrar_lote()     flujo por lotes
                     │
                     └─ RegistroAsistencia (PostgreSQL)
```

---

# Parte A — `core/utils_facial.py` (72 líneas)

**Rol:** concentra el preprocesado de imagen que comparten el registro y el motor.
Si el registro y la inferencia no usaran exactamente lo mismo, los vectores no
serían comparables.

**Dependencias:** `cv2`, `numpy`. **No importa Django.**

## A.1 `aplicar_clahe()` (líneas 5-16)

```python
def aplicar_clahe(imagen):
    if imagen is None or imagen.size == 0:
        return imagen
    try:
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ecualizado = clahe.apply(l)
        lab_mejorado = cv2.merge((l_ecualizado, a, b))
        return cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)
    except Exception:
        return imagen
```

- **Línea 6-7:** si la imagen es `None` o está vacía, la devuelve tal cual. Evita
  que OpenCV lance una excepción más adelante.
- **Línea 8:** el bloque `try` protege todo el proceso: si el formato de la imagen
  no es el esperado, se devuelve la original en lugar de romper el flujo.
- **Línea 9:** convierte de BGR (formato de OpenCV) a **CIELAB**. En LAB, el canal
  `L` es la luminosidad y `a`/`b` son el color. Así se puede ecualizar la luz sin
  alterar los colores de la cara.
- **Línea 10:** separa los tres canales.
- **Línea 11:** crea el objeto CLAHE con:
  - `clipLimit=2.0`: límite de amplificación del contraste. Cuanto más bajo, más
    suave; 2.0 evita amplificar el ruido.
  - `tileGridSize=(8, 8)`: la imagen se divide en 8×8 regiones y cada una se
    ecualiza por separado, lo que conserva el detalle local.
- **Línea 12:** aplica CLAHE solo al canal de luminosidad.
- **Línea 13:** vuelve a unir `L` ecualizado con `a` y `b` originales.
- **Línea 14:** convierte de LAB a BGR, el formato que espera el resto del
  pipeline.
- **Línea 15-16:** si algo falla, devuelve la imagen original sin modificar.

**Qué es CLAHE:** *Contrast Limited Adaptive Histogram Equalization*. Es una
ecualización de histograma por regiones con un límite de contraste. Sirve para
que una cara a contraluz o en una zona oscura del salón tenga el mismo aspecto
que una bien iluminada.

**Fórmula (concepto):** para cada región se calcula su histograma, se recorta la
parte que supera `clipLimit` y se redistribuye; luego se interpola entre regiones
vecinas para evitar costuras. El resultado es una `L` con más detalle local.

**Dónde se usa:** registro (`registro_unificado.py`, `registro.py`,
`generar_db.py`) y motor (`embedding_facial`).

## A.2 `embedding_facial()` (líneas 19-22)

```python
def embedding_facial(recognizer, imagen, landmarks):
    """Embedding normalizado (512,) con el mismo preprocesado en registro e inferencia."""
    img_limpia = aplicar_clahe(imagen)
    return recognizer.get_normalized_embedding(img_limpia, landmarks).flatten()
```

- **Línea 21:** aplica CLAHE a la imagen completa (no solo al recorte). El
  `recognizer` se encarga de alinear y recortar usando los landmarks.
- **Línea 22:** `get_normalized_embedding` devuelve un vector de 512 ya
  normalizado (norma 1). `.flatten()` garantiza forma `(512,)` en lugar de
  `(1, 512)`.

**Por qué es la pieza clave:** el registro hace exactamente
`aplicar_clahe(...)` + `get_normalized_embedding(...)` (ver
`registro_unificado.py`). El motor usa esta envoltura para no repetir el código y
para que un cambio de preprocesado se aplique a ambos lados.

## A.3 `es_imagen_borrosa()` (líneas 25-30)

```python
def es_imagen_borrosa(imagen, umbral=80):
    if imagen is None or imagen.size == 0:
        return True
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < umbral
```

- **Línea 26-27:** una imagen vacía se considera borrosa (no sirve).
- **Línea 28:** pasa a escala de grises; la nitidez se mide sobre luminancia.
- **Línea 29:** calcula el **Laplaciano** y su **varianza**.
- **Línea 30:** si la varianza es menor que el umbral, está borrosa.

**Fórmula:**

```
∇²I(x, y) = ∂²I/∂x² + ∂²I/∂y²          (Laplaciano)
score     = Var(∇²I) = (1/N) Σ (Lᵢ − L̄)²
```

Una imagen nítida tiene bordes fuertes y, por tanto, mucha varianza en el
Laplaciano; una imagen movida o desenfocada tiene bordes suaves y poca varianza.

**Umbral:** el valor por defecto es **80**, pero el registro usa **60**
(`self.umbral_blur = 60` en `registro_unificado.py`). Es un parámetro, no una
constante global.

**Dónde se usa:** solo en el registro. El motor no descarta frames por nitidez:
prefiere decidir con el consenso temporal.

## A.4 `alinear_rostro()` (líneas 33-72)

```python
def alinear_rostro(frame, landmarks_5pts, target_size=(224, 224)):
    if landmarks_5pts is None or len(landmarks_5pts) < 2:
        return None

    ojo_izq = landmarks_5pts[0]
    ojo_der = landmarks_5pts[1]

    dx = float(ojo_der[0] - ojo_izq[0])
    dy = float(ojo_der[1] - ojo_izq[1])

    angulo = np.degrees(np.arctan2(dy, dx))
    centro_ojos = (
        int((ojo_izq[0] + ojo_der[0]) / 2),
        int((ojo_izq[1] + ojo_der[1]) / 2),
    )

    dist_ojos = np.sqrt(dx**2 + dy**2)
    if dist_ojos < 10:
        return None

    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
    frame_rotado = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

    ancho_crop = int(dist_ojos * 4.5)
    alto_crop = int(dist_ojos * 5.5)

    start_x = max(0, int(centro_ojos[0] - ancho_crop // 2))
    start_y = max(0, int(centro_ojos[1] - alto_crop * 0.4))
    end_x = min(w, start_x + ancho_crop)
    end_y = min(h, start_y + alto_crop)

    rostro_alineado = frame_rotado[start_y:end_y, start_x:end_x]

    if rostro_alineado.size > 0 and rostro_alineado.shape[0] > 50:
        try:
            return cv2.resize(rostro_alineado, target_size)
        except Exception:
            return None
    return None
```

- **Línea 34-35:** si no hay al menos dos landmarks (los dos ojos), no se puede
  alinear; devuelve `None`.
- **Línea 37-38:** toma los dos primeros puntos como ojos (el orden que devuelve
  RetinaFace).
- **Línea 40-41:** diferencia horizontal y vertical entre los ojos.
- **Línea 43:** ángulo de inclinación de la línea de los ojos:
  `θ = atan2(dy, dx)` en grados. Si los ojos están a la misma altura, θ ≈ 0.
- **Línea 44-47:** punto medio entre los ojos; será el centro de rotación.
- **Línea 49:** distancia entre ojos:
  `d = √(dx² + dy²)`. Es la escala de la cara: más lejos, `d` más pequeña.
- **Línea 50-51:** si `d < 10` px, la cara es demasiado pequeña para alinear con
  fiabilidad; devuelve `None`.
- **Línea 53:** alto y ancho del frame.
- **Línea 54:** matriz de rotación de OpenCV alrededor del centro de los ojos:
  `M = [[α, β, (1−α)cx − βcy], [−β, α, βcx + (1−α)cy]]`, con
  `α = cos θ`, `β = sin θ`. Escala 1.0 (no cambia el tamaño).
- **Línea 55:** aplica la rotación con interpolación cúbica (`INTER_CUBIC`), que
  conserva mejor el detalle que la bilineal.
- **Línea 57-58:** tamaño del recorte proporcional a la distancia entre ojos:
  ancho `4.5·d`, alto `5.5·d`. Son proporciones típicas de una cara.
- **Línea 60:** esquina izquierda, centrada horizontalmente en los ojos.
- **Línea 61:** esquina superior. Se deja un 40 % del alto por encima de los ojos
  (frente) y el resto por debajo (nariz, boca, mentón).
- **Línea 62-63:** límites derechos e inferiores, recortados al tamaño del frame.
- **Línea 65:** recorta la región del rostro ya rotado.
- **Línea 67:** comprueba que el recorte no esté vacío y que tenga más de 50 px de
  alto.
- **Línea 68-71:** redimensiona a `target_size` (224×224 por defecto). Si falla,
  devuelve `None`.
- **Línea 72:** si no cumple las condiciones, `None`.

**Dónde se usa:** solo en el registro (`registro_unificado.py`, `registro.py`).
El motor no llama a esta función: `recognizer.get_normalized_embedding` hace su
propia alineación interna a partir de los landmarks.

## A.5 Quién usa cada función

| Función | Registro | Motor | Otros |
|---|---|---|---|
| `aplicar_clahe` | Sí | Sí (vía `embedding_facial`) | `generar_db.py` |
| `embedding_facial` | No (usa las dos llamadas) | **Sí** | — |
| `es_imagen_borrosa` | Sí (umbral 60) | No | — |
| `alinear_rostro` | Sí | No | — |

**Invariante:** si se cambia `aplicar_clahe` o la forma de calcular el embedding,
hay que **volver a registrar a los 9 estudiantes**. Los centroides viejos y los
nuevos dejarían de ser comparables.

---

# Parte B — `core/intrusos.py` (241 líneas)

**Rol:** decidir cuándo un desconocido es un intruso real y deduplicar por asiento.

**Dependencias:** solo `math`. Es lógica pura (sin Django) para poder probarla con
`pytest -m unit`.

> [!WARNING]
> Esta sección se actualizó el **2026-11-05** tras cambiar la regla (promedio en vez
> de máximo, deduplicación espacial y tamaño mínimo de 26 px). La calibración con
> los videos 1 y 3 está en `docs/analisis/unc/11_intrusos.md`.

## B.1 Constantes (líneas 17-31)

```python
import math

# Umbral sobre la similitud PROMEDIO de la pista para descartarla como posible
# estudiante registrado no confirmado (cuando el candidato no está ya anclado
# simultáneamente en otra pista activa del aula).
UMBRAL_INTRUSO = 0.24
# Frames consecutivos que debe sostener la pista para crear la fila.
FRAMES_INTRUSO = 15
# Frames sin ver la pista para limpiar su estado interno.
OLVIDO_PISTA = 30
# Ventana (en frames) y factor de distancia (en múltiplos del ancho del rostro)
# para reutilizar la misma fila de INTRUSO cuando ByteTrack reasigna track_id
# en el mismo asiento.
OLVIDO_ESPACIAL = 240
FACTOR_DISTANCIA_ASIENTO = 2.5
```

- **Línea 22 `UMBRAL_INTRUSO = 0.24`:** se compara contra el **promedio** de la
  pista, no contra su mejor frame.
- **Línea 24 `FRAMES_INTRUSO = 15`:** muestras de desconocido que debe acumular la
  pista.
- **Línea 26 `OLVIDO_PISTA = 30`:** sin ver la pista, se limpia su estado.
- **Línea 30 `OLVIDO_ESPACIAL = 240`:** ventana (~34 s a 7 fps) para reutilizar la
  fila de un intruso cuyo rastro se perdió.
- **Línea 31 `FACTOR_DISTANCIA_ASIENTO = 2.5`:** radio de reutilización, en
  múltiplos del ancho de la cara.

## B.2 Estado (líneas 34-67)

```python
class SeguimientoIntrusos:
    def __init__(
        self,
        umbral=UMBRAL_INTRUSO,
        frames=FRAMES_INTRUSO,
        olvido=OLVIDO_PISTA,
        olvido_espacial=OLVIDO_ESPACIAL,
        factor_distancia=FACTOR_DISTANCIA_ASIENTO,
    ):
        ...
        self._pistas = {}    # track_id -> {best, suma, muestras, rachas, ...}
        self._asientos = {}  # registro_id -> {bbox, ultimo, track_id, max_ancho}
```

Dos diccionarios: el **estado por pista** (estadísticas de similitud) y el
**mapa de asientos** (qué fila de intruso ocupa cada zona de la imagen).

## B.3 `_buscar_asiento_cercano()` (líneas 69-95)

```python
    def _buscar_asiento_cercano(self, bbox, vistos):
        """Busca si la caja cae en el asiento de un intruso perdido recientemente."""
        ...
        for reg_id, asiento in self._asientos.items():
            if asiento["track_id"] in vistos:
                continue
            if self.frame_idx - asiento["ultimo"] > self.olvido_espacial:
                continue
            ...
            dist = math.hypot(cx - scx, cy - scy)
            limite = max(ancho, sancho) * self.factor_distancia
            if dist <= limite and dist < mejor_dist:
                mejor_dist = dist
                mejor_id = reg_id
        return mejor_id
```

- Ignora los asientos cuya pista **sigue visible** (no es la misma persona).
- Ignora los asientos cuya última aparición fue hace más de `olvido_espacial`.
- Compara **centros** y exige `dist <= max(ancho_actual, ancho_guardado) × factor`.

## B.4 `evaluar()` (líneas 97-216)

```python
    def evaluar(self, decisiones):
        ...
        for decision in decisiones:
            ...
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

            ...
            candidato = decision.get("candidato")
            # Si el candidato top-1 ya está confirmado/anclado en otra pista de
            # este mismo frame, esta pista no puede ser esa persona.
            sim_efectiva = 0.0 if (candidato and candidato in confirmados_en_frame) else similitud
            info["suma"] += sim_efectiva
            info["muestras"] += 1
            promedio = info["suma"] / info["muestras"]

            if promedio >= self.umbral:
                info["descartada"] = True
                ...
                continue

            if (
                not info["descartada"]
                and info["rachas"] >= self.frames
                and info["registro_id"] is None
            ):
                reutilizar_id = self._buscar_asiento_cercano(bbox, vistos - {track_id})
                if reutilizar_id is not None:
                    decision["reutilizar_registro_id"] = reutilizar_id
                crear.append(decision)
```

Cuatro mecanismos:

1. **El promedio manda.** `sim_efectiva` es la similitud del frame, salvo que el
   candidato top-1 esté confirmado en **otra pista del mismo frame**: en ese caso la
   muestra vale 0, porque dos personas distintas no pueden ser el mismo alumno.
2. **Si el promedio supera el umbral**, la pista deja de ser intrusa y su fila se
   borra (si la creó ella).
3. **Los conflictos de identidad** nunca cuentan.
4. **Deduplicación por asiento**: si la pista califica donde había un intruso
   reciente, se marca `reutilizar_registro_id` en vez de crear otra fila.

## B.5 `marcar_registrado()` (líneas 218-241)

```python
    def marcar_registrado(self, track_id, registro_id, bbox=None, es_reutilizado=False):
        """Asocia la fila creada o reutilizada a la pista y registra su asiento."""
        info = self._pistas.get(track_id)
        if info is not None:
            info["registro_id"] = registro_id
            info["es_creador"] = not es_reutilizado
        ...
        asiento = self._asientos.get(registro_id)
        if asiento is None:
            self._asientos[registro_id] = {
                "bbox": bbox, "ultimo": self.frame_idx, "track_id": track_id, "max_ancho": ancho,
            }
        else:
            ...
```

`es_reutilizado=False` marca las filas reutilizadas: **no se borran** si la pista se
confirma después (la fila la creó otra pista).

## B.6 Invariantes

- El promedio es monotónico: una vez superado el umbral, la pista queda descartada.
- Una fila reutilizada nunca se borra por la confirmación de la pista que la usa.
- `_asientos` se limpia con las pistas (olvido de 30 frames) y al borrar la fila.

## B.7 Tests

`tests/test_registro.py::TestSeguimientoIntrusos` (8 pruebas): pico aislado, pista que
supera el umbral, descarte por mejora, conflicto, confirmación, pocos frames,
candidato confirmado en otra pista y **deduplicación espacial por asiento**.

# Parte C — `core/db_config.py` (132 líneas)

**Rol:** leer y escribir los centroides en PostgreSQL + pgvector desde scripts de
consola, **sin Django**. Los scripts de `scripts/` y `experiments/` deben poder
ejecutarse desde la terminal sin configurar el framework.

**Dependencias:** `psycopg2` (importado dentro de `conectar`), `numpy`.

## C.1 Conexión (líneas 14-20)

```python
CONEXION = {
    "host": os.environ.get("CV_DB_HOST", "localhost"),
    "port": int(os.environ.get("CV_DB_PORT", "5432")),
    "dbname": os.environ.get("CV_DB_NAME", "cv_college_db"),
    "user": os.environ.get("CV_DB_USER", "cv_college_user"),
    "password": os.environ.get("CV_DB_PASSWORD", "cv_college_2026"),
}
```

- **Línea 15:** host, por defecto `localhost`.
- **Línea 16:** puerto, convertido a entero.
- **Línea 17:** nombre de la base.
- **Línea 18:** usuario.
- **Línea 19:** contraseña. Se lee de `CV_DB_PASSWORD`; el valor por defecto es el
  de desarrollo. En producción hay que definir la variable.

## C.2 `conectar()` y `alias_de()` (líneas 25-39)

```python
def conectar():
    import psycopg2

    return psycopg2.connect(**CONEXION)


def alias_de(username, first_name=""):
    """Clave corta que usa el motor ('alexa', 'bob'...).

    Se toma del username (`unc_alexa` -> `alexa`) para conservar los nombres que
    ya usan los experimentos; si no tiene el prefijo, cae al nombre de pila.
    """
    if username and username.startswith(PREFIJO_ALIAS):
        return username[len(PREFIJO_ALIAS):]
    return (first_name or username or "").lower()
```

- **Línea 26:** import perezoso de `psycopg2`. Así el módulo se puede importar sin
  la librería instalada (aunque `conectar` fallaría).
- **Línea 28:** abre la conexión con el diccionario. `**CONEXION` pasa los cinco
  parámetros como nombrados.
- **Línea 37-38:** si el usuario empieza por `unc_`, devuelve lo que sigue:
  `unc_alexa` → `alexa`. Es la clave que usan el motor y FAISS.
- **Línea 39:** si no tiene el prefijo, usa el nombre de pila en minúsculas o el
  username.

## C.3 Formato de vector (líneas 42-51)

```python
def _parse_vector(texto):
    """Convierte el formato de pgvector '[0.1,0.2,...]' a np.ndarray float32."""
    if texto is None:
        return None
    limpio = texto.strip().lstrip("[").rstrip("]")
    return np.fromstring(limpio, sep=",", dtype=np.float32)


def _formatear_vector(vector):
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"
```

- **Línea 44-45:** si el texto es `None`, no hay vector.
- **Línea 46:** quita espacios y los corchetes del formato de pgvector:
  `"[0.1,0.2]"` → `"0.1,0.2"`.
- **Línea 47:** parsea los números separados por coma a `float32`.
- **Línea 51:** formatea un vector como texto pgvector con **8 decimales**:
  `"[0.12345678,...]"`. La precisión es suficiente para 512 dimensiones y evita
  problemas de redondeo en la comparación.

## C.4 `cargar_centroides()` (líneas 54-75)

```python
def cargar_centroides():
    """Devuelve {alias: vector 512d} leyendo attendance_faceembedding."""
    with conectar() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.username, u.first_name, e.embedding::text
                FROM attendance_faceembedding e
                JOIN users_user u ON u.id = e.user_id
                ORDER BY u.username
                """
            )
            filas = cur.fetchall()

    centrides = {}
    for username, first_name, vector in filas:
        vec = _parse_vector(vector)
        if vec is None or vec.shape[0] != 512:
            print(f"[AVISO] {username}: vector inválido, se omite")
            continue
        centrides[alias_de(username, first_name)] = vec
    return centrides
```

- **Línea 56:** abre conexión (se cierra sola al salir del `with`).
- **Línea 57:** cursor.
- **Línea 58-65:** SQL:
  - `e.embedding::text`: convierte el tipo `vector` a texto para parsearlo con
    numpy. Evita depender del adaptador de pgvector en Python.
  - `JOIN users_user`: une el embedding con su usuario para obtener el username.
  - `ORDER BY u.username`: orden estable, útil para reproducir índices.
- **Línea 66:** trae todas las filas.
- **Línea 68:** diccionario de salida.
- **Línea 69:** recorre usuario, nombre y vector.
- **Línea 70-73:** si el vector es inválido o no tiene 512 dimensiones, avisa y lo
  omite. No rompe el motor por una fila corrupta.
- **Línea 74:** guarda con la clave corta (`alexa`), no con `unc_alexa`.
- **Línea 75:** devuelve `{alias: np.ndarray(512)}`.

## C.5 `guardar_centroide()` (líneas 78-132)

```python
def guardar_centroide(alias, vector, modelo="arcface", n_muestras=0,
                      fuente="VIDEO", crear_usuario=True):
    """Inserta o actualiza el vector del alias. Devuelve True si guardó.

    `crear_usuario=False` se usa desde la web, donde el Usuario lo crea la API
    antes de guardar el embedding.
    """
    username = f"{PREFIJO_ALIAS}{alias}"
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.shape[0] != 512:
        raise ValueError(f"{alias}: se esperaba un vector de 512, llegó {vec.shape[0]}")
    norma = float(np.linalg.norm(vec))
    if norma == 0:
        raise ValueError(f"{alias}: vector nulo")
    vec = vec / norma

    with conectar() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users_user WHERE username = %s", (username,))
            fila = cur.fetchone()
            if fila is None:
                if not crear_usuario:
                    return False
                cur.execute(
                    """
                    INSERT INTO users_user
                        (password, is_superuser, username, first_name, last_name,
                         email, is_staff, is_active, date_joined, role, avatar)
                    VALUES ('', false, %s, %s, '', '', false, true, now(), 'STUDENT', NULL)
                    RETURNING id
                    """,
                    (username, alias.capitalize()),
                )
                user_id = cur.fetchone()[0]
                print(f"[BD] Usuario creado: {username}")
            else:
                user_id = fila[0]

            cur.execute(
                """
                INSERT INTO attendance_faceembedding
                    (user_id, embedding, modelo, n_muestras, fuente, fecha, created_at, updated_at)
                VALUES (%s, %s::vector, %s, %s, %s, now(), now(), now())
                ON CONFLICT (user_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    modelo = EXCLUDED.modelo,
                    n_muestras = EXCLUDED.n_muestras,
                    fuente = EXCLUDED.fuente,
                    fecha = EXCLUDED.fecha,
                    updated_at = now()
                """,
                (user_id, _formatear_vector(vec), modelo, int(n_muestras), fuente),
            )
        conn.commit()
    return True
```

- **Línea 85:** el username completo: `unc_` + alias.
- **Línea 86:** convierte a `float32` y aplana. Acepta listas o arrays.
- **Línea 87-88:** valida 512 dimensiones.
- **Línea 89-91:** calcula la norma; un vector nulo no se puede normalizar.
- **Línea 92:** normaliza `v / ‖v‖`.
- **Línea 94:** conexión.
- **Línea 96-97:** busca si el usuario ya existe.
- **Línea 98:** si no existe…
- **Línea 99-100:** y `crear_usuario=False`, no crea nada y devuelve `False`. La
  web usa este modo porque el usuario lo crea antes la API.
- **Línea 101-111:** inserta el usuario con rol `STUDENT`, `first_name` con el
  alias capitalizado y devuelve el id.
- **Línea 112:** aviso de usuario creado.
- **Línea 113-114:** si ya existía, usa su id.
- **Línea 116-130:** inserta o actualiza el embedding:
  - `ON CONFLICT (user_id) DO UPDATE`: la tabla tiene un vector por usuario;
    volver a registrar reemplaza el anterior.
  - `fecha = EXCLUDED.fecha`: actualiza la fecha del registro.
  - `updated_at = now()`.
- **Línea 131:** confirma la transacción. Sin `commit`, los cambios no se guardan.
- **Línea 132:** devuelve `True` si guardó.

### SQL de referencia

```sql
-- Leer todos los centroides
SELECT u.username, u.first_name, e.embedding::text
FROM attendance_faceembedding e
JOIN users_user u ON u.id = e.user_id
ORDER BY u.username;

-- Guardar (upsert)
INSERT INTO attendance_faceembedding
    (user_id, embedding, modelo, n_muestras, fuente, fecha, created_at, updated_at)
VALUES (%s, %s::vector, %s, %s, %s, now(), now(), now())
ON CONFLICT (user_id) DO UPDATE SET
    embedding = EXCLUDED.embedding,
    modelo = EXCLUDED.modelo,
    n_muestras = EXCLUDED.n_muestras,
    fuente = EXCLUDED.fuente,
    fecha = EXCLUDED.fecha,
    updated_at = now();
```

---

# Parte D — `core/registro_asistencia.py` (344 líneas)

**Rol:** traducir las decisiones del motor a filas de `RegistroAsistencia`.
Es la única capa que escribe asistencia; la usan el flujo en vivo y el batch.

**Dependencias:** Django (`ContentFile`, `timezone`), `cv2`, `uuid`, los modelos
`Estudiante` y `RegistroAsistencia`, y `UMBRAL_INTRUSO` de `core.intrusos`.

**Dos caminos:**

| Camino | Función | Cuándo |
|---|---|---|
| En vivo | `aplicar_decisiones()` | Cada frame de cámara o video |
| Lote | `acumular_decisiones()` + `registrar_lote()` | Carpeta de imágenes |

## D.1 `mapa_estudiantes()` (líneas 29-37)

```python
def mapa_estudiantes(asignacion):
    """Devuelve {alias_del_motor: Estudiante} para los inscritos en la clase."""
    from core.db_config import alias_de

    mapa = {}
    for inscripcion in asignacion.inscripciones_clase.select_related("estudiante__user"):
        usuario = inscripcion.estudiante.user
        mapa[alias_de(usuario.username, usuario.first_name)] = inscripcion.estudiante
    return mapa
```

- **Línea 31:** import perezoso de `alias_de` para no arrastrar `db_config` si no
  se usa.
- **Línea 34:** diccionario de salida.
- **Línea 35:** recorre las inscripciones de esa asignación. `select_related`
  trae el estudiante y su usuario en la misma consulta.
- **Línea 36-37:** extrae el alias (`unc_alexa` → `alexa`) y guarda el
  `Estudiante`.
- **Línea 38:** devuelve `{alias: Estudiante}`.

**Por qué importa:** el motor decide con alias, pero la base necesita el
`Estudiante`. Este mapa hace la traducción sin consultar por cada decisión.

## D.2 `_guardar_captura()` (líneas 40-53)

```python
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
```

- **Línea 41:** convierte la caja a enteros.
- **Línea 42:** alto y ancho del frame.
- **Línea 43:** recorta la cara **clampeando** las coordenadas al frame
  (`max(0, ...)` y `min(alto/ancho, ...)`). Sin esto, una caja parcialmente fuera
  rompería el recorte.
- **Línea 44-45:** si el recorte quedó vacío, no guarda nada.
- **Línea 46-47:** codifica el recorte en JPEG en memoria.
- **Línea 48-49:** si falla la codificación, no guarda.
- **Línea 50-54:** guarda el archivo con un nombre único
  (`captura_<8 hex>.jpg`) usando `ContentFile`, y `save=True` actualiza el campo
  `captura_fraude` del registro.
- **Línea 55:** fin.

**Dónde queda:** `media/fraudes/` (configurado en `MEDIA_ROOT`). La web lo sirve
en `DEBUG`.

## D.3 `_registrar_desconocido()` (líneas 56-96)

```python
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
```

- **Línea 66:** etiqueta **estable** de la fila: motivo + número de pista. No
  incluye la similitud, para que no cambie en cada frame.
- **Línea 67:** hora local (America/Bogota).
- **Línea 68-73:** busca una fila existente **por el prefijo** de la nota
  (`startswith`). Es lo que permite añadir la mejor coincidencia al crear sin
  romper la deduplicación: si se comparara la nota completa, cada frame con una
  similitud distinta crearía una fila nueva.
- **Línea 75:** si no existe…
- **Línea 76:** empieza con la etiqueta.
- **Línea 77-81:** añade ` Mejor coincidencia: <alias> (<sim>).` si hay candidato.
  Ese texto es lo que el profesor ve en `/sesiones/<id>/`.
- **Línea 82-91:** crea la fila con:
  - `estudiante=None` (no es un estudiante identificado),
  - `tipo_evento` (`INTRUSO` o `SPOOFING`),
  - `notas_auditoria` con el detalle,
  - `hora_entrada` y `ultima_vez_visto` con la hora actual,
  - `es_fraude` según el caso,
  - `similitud_ia` con el mejor score.
- **Línea 92:** guarda la captura del rostro.
- **Línea 93-97:** si ya existía, solo actualiza `ultima_vez_visto` y, si llega,
  la similitud. No crea duplicados.
- **Línea 98:** devuelve el registro (lo usa `aplicar_decisiones` para
  `marcar_registrado`).

## D.4 `aplicar_decisiones()` (líneas 99-200)

```python
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
            _registrar_desconocido(
                sesion, frame, bbox, es_fraude=True,
                tipo_evento=RegistroAsistencia.TipoEvento.SPOOFING,
                track_id=track_id, nota="Posible fraude (anti-spoofing).",
                candidato=decision.get("candidato"), similitud=similitud,
            )
            resumen["fraudes"] += 1
            continue

        if estado == "DESCONOCIDO":
            if intrusos_permitidos is not None and track_id not in intrusos_permitidos:
                continue
            ancho = int(bbox[2]) - int(bbox[0])
            if ancho >= TAMANO_MINIMO_INTRUSO:
                registro = _registrar_desconocido(
                    sesion, frame, bbox, es_fraude=False,
                    tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO,
                    track_id=track_id, nota="Persona no registrada.",
                    candidato=decision.get("candidato"), similitud=similitud,
                )
                if seguimiento is not None and registro is not None:
                    seguimiento.marcar_registrado(track_id, registro.id)
                resumen["intrusos"] += 1

    return resumen
```

### Orden de las reglas

El orden importa: cada decisión cae en la **primera** regla que cumpla.

| # | Condición | Qué escribe |
|---|---|---|
| 1 | Confirmado y alias inscrito | `ASISTENCIA_NORMAL` |
| 2 | Confirmado y alias del profesor | Nota en `sesion.notas_profesor` |
| 3 | Confirmado, no inscrito, pero existe `unc_<alias>` | `OTRA_SECCION` |
| 4 | `FRAUDE` | `SPOOFING` con `es_fraude=True` |
| 5 | `DESCONOCIDO` permitido y ancho ≥ 40 px | `INTRUSO` |

- **Línea 111-112:** hora actual y resumen que devuelve la función.
- **Línea 114-122:** si hay clasificador de intrusos:
  - `evaluar` devuelve las decisiones que califican y las filas a borrar.
  - `intrusos_permitidos` es el conjunto de `track_id` que sí pueden escribir.
  - Si hay filas a descartar, se **borran también sus capturas** del disco.
  - Cuando no hay clasificador (`seguimiento=None`), el comportamiento es el
    antiguo: una fila por pista. Los tests unitarios usan ese modo.
- **Línea 124-130:** descompone la decisión y marca si está confirmada.
- **Línea 132-150 — regla 1:** si el alias está en los inscritos:
  - `get_or_create` con `(sesion, estudiante)`: la tabla tiene
    `unique_together = ("sesion", "estudiante")`, así que una persona tiene una
    sola fila por sesión.
  - Si ya existía, actualiza `ultima_vez_visto` y la similitud.
  - Añade el alias a `resumen["confirmados"]` (la web lo muestra en vivo).
- **Línea 152-157 — regla 2:** si el confirmado es el profesor, no se crea fila
  de asistencia; se añade una nota a la sesión (una sola vez).
- **Línea 159-175 — regla 3:** un registrado que no está inscrito en esa sección.
  - Reconstruye el username `unc_<alias>`.
  - Si existe el estudiante, crea una fila `OTRA_SECCION`.
  - No actualiza `ultima_vez_visto` si ya existía (solo `get_or_create`).
- **Línea 177-185 — regla 4:** `FRAUDE` → fila `SPOOFING` con captura. La nota
  incluye la mejor coincidencia.
- **Línea 187-200 — regla 5:** `DESCONOCIDO`.
  - Si hay clasificador y la pista no está permitida, se ignora.
  - Si el ancho es menor de 40 px, no se guarda captura ni fila (es una cara
    demasiado pequeña para identificar).
  - Crea la fila y avisa al clasificador con `marcar_registrado`.
  - Suma al resumen.
- **Línea 202:** devuelve el resumen.

### Qué devuelve el resumen

```python
{
    "confirmados": ["alexa", "bob"],  # alias confirmados en este frame
    "intrusos": 1,                    # filas de intruso creadas
    "fraudes": 0,                     # filas de spoofing creadas
}
```

`core.servicio_en_vivo` lo usa para el contador y la lista de confirmados de la
página `/en-vivo/`.

## D.5 Flujo por lotes

### `reiniciar_pistas()` (líneas 207-215)

```python
def reiniciar_pistas(motor):
    """Deja el motor listo para una imagen nueva sin recargar los modelos."""
    import supervision as sv

    motor.frame_idx = 0
    motor.identidades_ancladas = {}
    motor.votos_identidad = {}
    motor.vetos_identidad = {}
    motor.tracker = sv.ByteTrack()
```

- **Línea 209:** import local de `supervision`.
- **Línea 211:** reinicia el contador de frames.
- **Línea 212-214:** limpia anclas, votos y vetos.
- **Línea 215:** crea un ByteTrack nuevo. Cada imagen es independiente: no se
  arrastran pistas de la anterior.

### `nuevo_acumulado()` (líneas 218-223)

```python
def nuevo_acumulado():
    return {
        "confirmados": {},   # alias -> {n, sim_max, frame, bbox, primer_indice}
        "intrusos": {"n": 0, "frame": None, "bbox": None, "primer_indice": None},
        "fraudes": {"n": 0, "frame": None, "bbox": None, "primer_indice": None},
    }
```

- **Línea 220:** diccionario de confirmados por alias. Guarda cuántas imágenes lo
  confirmaron, la mejor similitud y el frame/bbox de esa mejor evidencia.
- **Línea 221:** contador de detecciones de intruso y la primera evidencia.
- **Línea 222:** lo mismo para fraude.

### `acumular_decisiones()` (líneas 226-268)

```python
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
```

- **Línea 228:** recorre las decisiones de una imagen.
- **Línea 236-247 — confirmado:** acumula por alias:
  - `n`: número de imágenes en que apareció.
  - `sim_max`: la mejor similitud vista.
  - `frame`/`bbox`: la imagen y la caja de esa mejor similitud (para la captura).
  - `primer_indice`: en qué imagen apareció por primera vez.
- **Línea 249-250 — fraude:** cuenta y guarda la primera evidencia.
- **Línea 251-261 — desconocido:**
  - descarta caras de menos de 40 px de ancho,
  - descarta similitudes ≥ 0.18 (probablemente un registrado en mal frame),
  - descarta conflictos de identidad,
  - si pasa, cuenta como intruso.
- **Línea 263:** ignora `VERIFICANDO`, `CENTRATE`, `ERROR`.
- **Línea 265-269:** suma la detección y, si es la primera, guarda frame/bbox.

### `registrar_lote()` (líneas 270-344)

```python
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
```

- **Línea 279:** hora actual.
- **Línea 280:** lista de alias efectivamente registrados.
- **Línea 282:** recorre los confirmados.
- **Línea 283-284:** exige al menos `minimo_imagenes` (3) confirmaciones. En un
  lote de menos de 3 imágenes, el procesador pasa `minimo_imagenes = total`.
- **Línea 286-299:** si el alias está inscrito, `update_or_create` con
  `ASISTENCIA_NORMAL`. Usa `sim_max` (la mejor evidencia) y lo añade a
  `registrados`.
- **Línea 301-306:** si es el profesor, nota en la sesión.
- **Línea 308-320:** si no está inscrito pero existe el estudiante, `OTRA_SECCION`.
- **Línea 322-332:** intrusos: **una sola fila** por lote con el número de
  detecciones en la nota. Se guarda la captura del primer frame con evidencia.
- **Línea 334-344:** fraudes: una sola fila por lote, `es_fraude=True`, con
  captura.
- **Línea 346:** devuelve los alias registrados (para el log del batch).

## D.6 Decisión → fila

| Decisión del motor | Condición | Fila creada |
|---|---|---|
| `CONFIRMADO` / `ANCLADO` | alias inscrito | `ASISTENCIA_NORMAL` |
| `CONFIRMADO` / `ANCLADO` | alias del profesor | nota en la sesión |
| `CONFIRMADO` / `ANCLADO` | no inscrito, existe `unc_<alias>` | `OTRA_SECCION` |
| `FRAUDE` | — | `SPOOFING`, `es_fraude=True` |
| `DESCONOCIDO` | pista permitida y ancho ≥ 40 px | `INTRUSO`, `es_fraude=False` |
| `VERIFICANDO` / `CENTRATE` / `ERROR` | — | nada |

## D.7 Casos límite y decisiones de diseño

- **Una fila por estudiante y sesión:** lo impone `unique_together`. Por eso se
  usa `get_or_create`/`update_or_create` y no `create`.
- **Deduplicación de intrusos:** la nota contiene `Pista <track_id>`; el filtro
  por `startswith` permite añadir la mejor coincidencia sin crear filas nuevas.
- **Borrado de filas descartadas:** si una pista se confirma después, se borra la
  fila de intruso **y su captura** para no dejar basura en `media/`.
- **Captura solo al crear:** no se reescribe la imagen en cada frame.
- **Batch vs vivo:** el batch exige 3 imágenes; el vivo exige 3 votos
  consecutivos. Son consensos distintos porque la unidad de observación es
  distinta (imagen vs frame).
- **El batch no usa `SeguimientoIntrusos`:** no hay tracking continuo entre
  imágenes; se filtra por similitud (< 0.18) y por conflictos.

## D.8 Tests

- `tests/test_registro.py::TestSeguimientoIntrusos` (6 pruebas del clasificador).
- `apps/attendance/tests.py` (32 pruebas Django):
  - `test_estudiante_confirmado_registra_asistencia`,
  - `test_desconocido_es_intruso_no_fraude`,
  - `test_fraude_marca_es_fraude`,
  - `test_lote_exige_evidencia_en_tres_imagenes`,
  - `test_lote_agrupa_intrusos_en_una_fila`,
  - y las de sesiones/descartar.

---

## Cierre del capítulo

| Archivo | Qué se llevó |
|---|---|
| `utils_facial.py` | El preprocesado que hace comparables registro e inferencia |
| `intrusos.py` | La lógica que evita los intrusos falsos |
| `db_config.py` | El puente entre los scripts y pgvector |
| `registro_asistencia.py` | La traducción de decisiones a filas y los dos flujos |

**Siguiente capítulo (3):** `core/procesador_batch.py`,
`core/servicio_en_vivo.py`, `core/sesion_demo.py` y `scripts/registro_video.py`
/ `registro_unificado.py`. Está en `docs/motor_3.md`.

---

## Referencias

| Tema | Documento |
|---|---|
| Capítulo 1 (motor) | `docs/motor.md` |
| Intrusos (medición) | `docs/analisis/unc/11_intrusos.md` |
| Base vectorial | `docs/base_datos.md` |
| Cómputo | `docs/computo.md` |
| Comandos | `docs/comandos.md` |
