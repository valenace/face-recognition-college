import pytest
import numpy as np
import faiss


@pytest.mark.unit
class TestFAISSSearch:
    def _build_index(self, db):
        nombres = list(db.keys())
        vectores = np.array(list(db.values())).astype("float32")
        index = faiss.IndexFlatIP(vectores.shape[1])
        index.add(vectores)
        return nombres, index

    def test_misma_persona_alta_similitud(self):
        np.random.seed(42)
        base = np.random.randn(512).astype("float32")
        base /= np.linalg.norm(base)
        query = base + np.random.randn(512).astype("float32") * 0.02
        query /= np.linalg.norm(query)
        nombres, index = self._build_index({"user1": base})
        dist, idx = index.search(np.array([query]).astype("float32"), 1)
        assert dist[0][0] > 0.7

    def test_diferente_persona_baja_similitud(self):
        np.random.seed(42)
        v1 = np.random.randn(512).astype("float32")
        v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(512).astype("float32")
        v2 /= np.linalg.norm(v2)
        nombres, index = self._build_index({"user1": v1})
        dist, idx = index.search(np.array([v2]).astype("float32"), 1)
        assert dist[0][0] < 0.5

    def test_multiple_personas_retorna_correcto(self):
        np.random.seed(42)
        lino_vec = np.random.randn(512).astype("float32")
        lino_vec /= np.linalg.norm(lino_vec)
        jorge_vec = np.random.randn(512).astype("float32")
        jorge_vec /= np.linalg.norm(jorge_vec)
        valu_vec = np.random.randn(512).astype("float32")
        valu_vec /= np.linalg.norm(valu_vec)
        db = {"Lino": lino_vec, "Jorge": jorge_vec, "Valu": valu_vec}
        nombres, index = self._build_index(db)
        lino_query = lino_vec + np.random.randn(512).astype("float32") * 0.01
        lino_query /= np.linalg.norm(lino_query)
        dist, idx = index.search(np.array([lino_query]).astype("float32"), 1)
        assert nombres[idx[0][0]] == "Lino"
        assert dist[0][0] > 0.7

    def test_embedding_dim_512(self):
        vec = np.random.randn(512).astype("float32")
        vec /= np.linalg.norm(vec)
        index = faiss.IndexFlatIP(512)
        index.add(np.array([vec]))
        assert index.d == 512
        assert index.ntotal == 1


@pytest.mark.integration
@pytest.mark.slow
class TestFAISSConModelosReales:
    def test_embeddings_reales_faiss(self, recognizer, detector,
                                     sample_face_jorge, sample_face_lino,
                                     sample_face_valu):
        rostros_j = detector.detect(sample_face_jorge)
        rostros_l = detector.detect(sample_face_lino)
        rostros_v = detector.detect(sample_face_valu)

        if not rostros_j or not rostros_l or not rostros_v:
            pytest.skip("No se detectaron rostros en las imagenes del dataset")

        emb_j = recognizer.get_normalized_embedding(sample_face_jorge, rostros_j[0].landmarks)
        emb_l = recognizer.get_normalized_embedding(sample_face_lino, rostros_l[0].landmarks)
        emb_v = recognizer.get_normalized_embedding(sample_face_valu, rostros_v[0].landmarks)

        db = {"Jorge": emb_j, "Lino": emb_l, "Valu": emb_v}
        nombres = list(db.keys())
        vectores = np.array(list(db.values())).astype("float32")
        index = faiss.IndexFlatIP(512)
        index.add(vectores)

        dist, idx = index.search(np.array([emb_l]).astype("float32"), 1)
        assert nombres[idx[0][0]] == "Lino"
        assert dist[0][0] > 0.9

    def test_embeddings_cross_person_no_match(self, recognizer, detector,
                                               sample_face_jorge, sample_face_lino):
        rostros_j = detector.detect(sample_face_jorge)
        rostros_l = detector.detect(sample_face_lino)
        if not rostros_j or not rostros_l:
            pytest.skip("No se detectaron rostros en las imagenes del dataset")

        emb_j = recognizer.get_normalized_embedding(sample_face_jorge, rostros_j[0].landmarks)
        emb_l = recognizer.get_normalized_embedding(sample_face_lino, rostros_l[0].landmarks)

        db = {"Lino": emb_l}
        nombres = list(db.keys())
        vectores = np.array(list(db.values())).astype("float32")
        index = faiss.IndexFlatIP(512)
        index.add(vectores)

        dist, idx = index.search(np.array([emb_j]).astype("float32"), 1)
        assert dist[0][0] < 0.5
