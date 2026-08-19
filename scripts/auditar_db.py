import pickle
from pathlib import Path

DB_FILE = Path("data/database_embeddings.pkl")


def auditar_db(db_path=DB_FILE):
    """
    Carga la DB de embeddings e imprime un resumen.

    Returns:
        dict: {nombre: embedding} o None si el archivo no existe.
    """
    if not Path(db_path).exists():
        print("La base de datos aún no existe o está vacía.")
        return None

    with open(db_path, "rb") as f:
        db = pickle.load(f)

    print(f"\n📊 TOTAL DE ESTUDIANTES REGISTRADOS: {len(db)}")
    print("-" * 40)
    for i, nombre in enumerate(db.keys(), 1):
        print(f" {i}. {nombre}")

    return db


if __name__ == "__main__":
    auditar_db()