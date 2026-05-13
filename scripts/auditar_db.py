import pickle
from pathlib import Path

db_path = Path("data/database_embeddings.pkl")

if db_path.exists():
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    print(f"\n📊 TOTAL DE ESTUDIANTES REGISTRADOS: {len(db)}")
    print("-" * 40)
    for i, nombre in enumerate(db.keys(), 1):
        print(f" {i}. {nombre}")
else:
    print("❌ La base de datos aún no existe o está vacía.")
