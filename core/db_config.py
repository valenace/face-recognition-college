"""
Acceso a la base vectorial (PostgreSQL + pgvector) desde los scripts de consola.

No importa Django a propósito: los scripts de `scripts/` y `experiments/` deben
poder ejecutarse desde la terminal. La configuración se puede sobrescribir con
las variables de entorno CV_DB_HOST, CV_DB_PORT, CV_DB_NAME, CV_DB_USER y
CV_DB_PASSWORD.
"""

import os

import numpy as np

CONEXION = {
    "host": os.environ.get("CV_DB_HOST", "localhost"),
    "port": int(os.environ.get("CV_DB_PORT", "5432")),
    "dbname": os.environ.get("CV_DB_NAME", "cv_college_db"),
    "user": os.environ.get("CV_DB_USER", "cv_college_user"),
    "password": os.environ.get("CV_DB_PASSWORD", "cv_college_2026"),
}

PREFIJO_ALIAS = "unc_"


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
    return (username or first_name or "").lower()


def _parse_vector(texto):
    """Convierte el formato de pgvector '[0.1,0.2,...]' a np.ndarray float32."""
    if texto is None:
        return None
    limpio = texto.strip().lstrip("[").rstrip("]")
    return np.fromstring(limpio, sep=",", dtype=np.float32)


def _formatear_vector(vector):
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"


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
