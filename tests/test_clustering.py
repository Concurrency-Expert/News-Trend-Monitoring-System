"""Unit and integration tests for Phase 2 Indonesian NLP & topic clustering."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from src.analytics import (
    ClusteringResult,
    IncrementalTfidfVectorizer,
    IndonesianTextPreprocessor,
    TopicClusterer,
)


def _make_article(
    url_hash: str,
    title: str,
    clean_text: str,
    source: str = "detik",
    published_at: datetime | str | None = None,
    topic_id: str | None = None,
) -> dict:
    return {
        "url_hash": url_hash,
        "title": title,
        "clean_text": clean_text,
        "url": f"https://example.com/{source}/{url_hash}",
        "source": source,
        "published_at": published_at or datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc),
        "topic_id": topic_id,
        "created_at": datetime(2026, 11, 19, 6, 15, tzinfo=timezone.utc),
    }


def test_preprocessor_strips_portal_datelines_and_media_stopwords():
    # Arrange
    prep = IndonesianTextPreprocessor(use_stemming=True)

    detik_title = "Komisi III DPR Ungkap Alasan UU Perampasan Aset Tak Atur Illicit Enrichment"
    detik_body = "Jakarta - Ketua Komisi III DPR membahas rancangan perampasan aset tindak pidana."
    republika_title = "Kementan Sebut Kemitraan Sawit Kunci Dongkrak Produktivitas Nasional |Republika Online"
    republika_body = "REPUBLIKA.CO.ID, JAKARTA, — Kementerian Pertanian menegaskan kemitraan perkebunan sawit."
    cnbc_title = "Bos IMF Ungkap Efek AI ke Ekonomi Global Mulai Terasa Tahun Ini"
    cnbc_body = "Jakarta, CNBC Indonesia - Direktur Pelaksana IMF menjelaskan dampak teknologi kecerdasan buatan."
    kompas_title = "IHSG Ditutup Merah di Sesi I Imbas Melemahnya Saham Big Caps"
    kompas_body = (
        "Tumbuh bersama kekuatan mimpi perempuan Indonesia\n\n"
        "Kompas.com, 19 November 2026, 13:08 WIB\n\n"
        "JAKARTA, KOMPAS.com- Indeks Harga Saham Gabungan melemah akibat saham perbankan."
    )

    # Act
    detik_out = prep.preprocess(f"{detik_title}\n{detik_body}")
    republika_out = prep.preprocess(f"{republika_title}\n{republika_body}")
    cnbc_out = prep.preprocess(f"{cnbc_title}\n{cnbc_body}")
    kompas_out = prep.preprocess(f"{kompas_title}\n{kompas_body}")

    # Assert: portal brands, datelines, temporal & attribution noise are absent
    for combined in [detik_out, republika_out, cnbc_out, kompas_out]:
        tokens = set(combined.split())
        assert tokens.isdisjoint(
            {"jakarta", "republika", "online", "cnbc", "kompas", "wib", "november", "ungkap", "sebut", "tegas", "jelas"}
        )

    assert "mimpi" not in kompas_out.split()
    assert "perempuan" not in kompas_out.split()
    assert "aset" in detik_out.split()
    assert "sawit" in republika_out.split()
    assert "ekonomi" in cnbc_out.split()
    assert "ihsg" in kompas_out.split()


def test_preprocessor_stemming_and_cache_consistency():
    # Arrange
    prep = IndonesianTextPreprocessor(use_stemming=True)

    # Act
    pairs_a = prep.tokenize_with_surfaces("Pelemahan saham perbankan")
    pairs_b = prep.tokenize_with_surfaces("Saham melemah tajam")
    empty_none = prep.preprocess(None)
    empty_nums = prep.preprocess("12345 999 --- !!!")

    # Assert
    stems_a = [term for term, _ in pairs_a]
    stems_b = [term for term, _ in pairs_b]
    assert stems_a[0] == stems_b[1] == "lemah"
    assert "lemah" in prep._stem_cache.values()
    assert empty_none == ""
    assert empty_nums == ""


def test_incremental_tfidf_dimension_alignment_and_serialization():
    # Arrange
    vec_v1 = IncrementalTfidfVectorizer(max_features=10, sublinear_tf=True)
    batch1 = ["ihsg saham bank melemah", "saham big caps terkoreksi"]

    # Act
    mat1 = vec_v1.fit_transform_batch(batch1)
    saved_state = vec_v1.export_state()
    old_centroid = np.mean(mat1, axis=0)
    old_centroid = old_centroid / np.linalg.norm(old_centroid)

    vec_v2 = IncrementalTfidfVectorizer.from_state(saved_state)
    batch2 = ["perampasan aset komisi dpr", "ihsg saham bank rebound"]
    mat2 = vec_v2.fit_transform_batch(batch2)
    aligned_old_centroid = vec_v2.align_vector(old_centroid)

    # Assert
    assert mat1.shape[0] == 2
    assert mat2.shape[0] == 2
    assert mat2.shape[1] >= mat1.shape[1]
    assert aligned_old_centroid.shape == (len(vec_v2.terms),)
    assert math.isclose(float(np.linalg.norm(aligned_old_centroid)), 1.0, rel_tol=1e-6)
    for row in mat2:
        assert math.isclose(float(np.linalg.norm(row)), 1.0, rel_tol=1e-6)
    # Batch 1 term positions remain intact after Batch 2 expansion
    for idx, term in enumerate(saved_state["terms"]):
        assert vec_v2.term_to_idx[term] == idx
    assert len(vec_v2.terms) <= 10


def test_centroid_matching_updates_active_topic_and_ignores_archived_and_assigned():
    # Arrange
    clusterer = TopicClusterer(similarity_threshold=0.60, title_weight=3)
    base_articles = [
        _make_article(
            "h01",
            "Komisi III DPR Bahas RUU Perampasan Aset",
            "Komisi III DPR membahas substansi RUU Perampasan Aset dan illicit enrichment.",
            source="detik",
            published_at=datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc),
        ),
        _make_article(
            "h02",
            "Pembahasan RUU Perampasan Aset di Komisi III DPR",
            "RUU Perampasan Aset serta aturan illicit enrichment dibahas Komisi III DPR.",
            source="kompas",
            published_at=datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc),
        ),
    ]
    res1 = clusterer.cluster_articles(
        base_articles,
        active_topics=[],
        batch_time=datetime(2026, 11, 19, 7, 0, tzinfo=timezone.utc),
    )
    assert len(res1.new_topics) == 1
    active_topic = res1.new_topics[0]

    archived_topic = {
        "topic_id": "topik_20261101_001",
        "label": "Kemitraan Sawit",
        "keywords": ["kemitraan", "sawit", "produktivitas", "nasional"],
        "centroid_vector": active_topic["centroid_vector"],
        "status": "archived",
        "article_count": 4,
        "first_seen_at": datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc),
        "last_updated_at": datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc),
    }

    batch2 = [
        _make_article(
            "h03",
            "DPR Matangkan RUU Perampasan Aset Bersama Komisi III",
            "Komisi III DPR mematangkan RUU Perampasan Aset mengenai illicit enrichment.",
            source="cnbc",
            published_at=datetime(2026, 11, 19, 9, 30, tzinfo=timezone.utc),
        ),
        _make_article(
            "h_already",
            "Komisi III DPR Bahas Perampasan Aset Lanjutan",
            "Artikel ini sudah memiliki topic_id sehingga wajib dilewati.",
            source="republika",
            topic_id="topik_existing_999",
        ),
    ]

    # Act
    res2 = clusterer.cluster_articles(
        batch2,
        active_topics=[active_topic, archived_topic],
        batch_time=datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc),
    )

    # Assert
    assert res2.article_topic_assignments == {"h03": active_topic["topic_id"]}
    assert "h_already" not in res2.article_topic_assignments
    assert "h_already" not in res2.unmatched_article_keys
    assert len(res2.updated_topics) == 1
    upd = res2.updated_topics[0]
    assert upd["topic_id"] == active_topic["topic_id"]
    assert upd["article_count"] == 3
    assert upd["last_updated_at"] == datetime(2026, 11, 19, 9, 30, tzinfo=timezone.utc)
    assert math.isclose(float(np.linalg.norm(upd["centroid_vector"])), 1.0, rel_tol=1e-6)
    assert res2.new_topics == []


def test_similarity_threshold_boundary():
    # Arrange: construct exact 2-term space where cosine similarity can be controlled
    vec_helper = IncrementalTfidfVectorizer(max_features=100, sublinear_tf=False)
    prep = IndonesianTextPreprocessor(use_stemming=False)
    clusterer = TopicClusterer(
        similarity_threshold=0.60,
        title_weight=1,
        preprocessor=prep,
        vectorizer=vec_helper,
    )

    # First register 3 orthogonal vocabulary tokens with uniform DF=1
    vec_helper.fit_transform_batch(["alpha beta", "alpha beta", "gamma"])
    c_vec = vec_helper.align_vector([1.0 if t in {"alpha", "beta"} else 0.0 for t in vec_helper.terms])

    active_topic = {
        "topic_id": "topik_20261122_001",
        "label": "Alpha Beta",
        "keywords": ["alpha", "beta"],
        "centroid_vector": c_vec.tolist(),
        "status": "active",
        "article_count": 2,
        "first_seen_at": datetime(2026, 11, 22, 1, 0, tzinfo=timezone.utc),
        "last_updated_at": datetime(2026, 11, 22, 1, 0, tzinfo=timezone.utc),
    }

    # Test below threshold (only shares 1 of 2 terms -> cos = 1/sqrt(2) ~ 0.707 vs 1 of 3 -> 1/sqrt(6) ~ 0.408)
    art_match = _make_article("h_match", "alpha", "alpha beta")
    art_miss = _make_article("h_miss", "alpha", "gamma delta epsilon")

    # Act
    res = clusterer.cluster_articles(
        [art_match, art_miss],
        active_topics=[active_topic],
        batch_time=datetime(2026, 11, 22, 2, 0, tzinfo=timezone.utc),
    )

    # Assert
    assert res.article_topic_assignments.get("h_match") == "topik_20261122_001"
    assert "h_miss" in res.unmatched_article_keys


def test_dynamic_discovery_creates_topics_with_deterministic_2word_label():
    # Arrange
    clusterer = TopicClusterer(similarity_threshold=0.60, min_cluster_size=2, title_weight=3)
    batch = [
        _make_article(
            "detik_aset",
            "Komisi III DPR Ungkap Alasan UU Perampasan Aset Tak Atur Illicit Enrichment",
            "Ketua Komisi III DPR membahas alasan UU Perampasan Aset tidak mengatur illicit enrichment.",
            source="detik",
            published_at=datetime(2026, 11, 19, 6, 14, tzinfo=timezone.utc),
        ),
        _make_article(
            "kompas_aset",
            "UU Perampasan Aset dan Illicit Enrichment Dibahas Komisi III DPR |Kompas.com",
            "JAKARTA, KOMPAS.com - Komisi III DPR mengkaji ketentuan UU Perampasan Aset terkait illicit enrichment.",
            source="kompas",
            published_at=datetime(2026, 11, 19, 6, 30, tzinfo=timezone.utc),
        ),
        _make_article(
            "kompas_ihsg",
            "IHSG Ditutup Merah di Sesi I Imbas Melemahnya Saham Big Caps",
            "Kompas.com, 19 November 2026, 13:08 WIB\n\nJAKARTA, KOMPAS.com- Indeks Harga Saham Gabungan melemah tertekan saham big caps.",
            source="kompas",
            published_at=datetime(2026, 11, 19, 6, 5, tzinfo=timezone.utc),
        ),
        _make_article(
            "cnbc_ihsg",
            "Saham Big Caps Tekan IHSG Hingga Ditutup Merah di Sesi I",
            "Jakarta, CNBC Indonesia - Penurunan saham big caps menekan pergerakan IHSG hingga sesi I berakhir merah.",
            source="cnbc",
            published_at=datetime(2026, 11, 19, 6, 20, tzinfo=timezone.utc),
        ),
        _make_article(
            "republika_sawit",
            "Kementan Sebut Kemitraan Sawit Kunci Dongkrak Produktivitas Nasional |Republika Online",
            "REPUBLIKA.CO.ID, JAKARTA, — Kementerian Pertanian menekankan kemitraan perkebunan sawit rakyat.",
            source="republika",
            published_at=datetime(2026, 11, 19, 7, 0, tzinfo=timezone.utc),
        ),
    ]

    # Act
    res = clusterer.cluster_articles(
        batch,
        active_topics=[],
        batch_time=datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc),
    )

    # Assert
    assert isinstance(res, ClusteringResult)
    assert len(res.new_topics) == 2
    assert [t["topic_id"] for t in res.new_topics] == [
        "topik_20261119_001",
        "topik_20261119_002",
    ]
    assert res.article_topic_assignments["kompas_ihsg"] == res.article_topic_assignments["cnbc_ihsg"]
    assert res.article_topic_assignments["detik_aset"] == res.article_topic_assignments["kompas_aset"]
    assert res.article_topic_assignments["kompas_ihsg"] != res.article_topic_assignments["detik_aset"]
    assert res.unmatched_article_keys == ["republika_sawit"]
    for topic in res.new_topics:
        assert len(topic["label"].split()) == 2
        for forbidden in ("Kompas", "Republika", "Detik", "Cnbc", "Jakarta"):
            assert forbidden not in topic["label"]


def test_multi_batch_lifecycle_and_singleton_promotion():
    # Arrange
    clusterer = TopicClusterer(similarity_threshold=0.60, min_cluster_size=2, title_weight=3)
    batch1 = [
        _make_article(
            "a1",
            "Pupuk Indonesia Siapkan Stok Pupuk Subsidi Petani",
            "Pupuk Indonesia menyiapkan stok pupuk subsidi untuk petani nasional.",
            source="cnbc",
            published_at=datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc),
        ),
        _make_article(
            "a2",
            "Stok Pupuk Subsidi Petani Disiapkan Pupuk Indonesia",
            "Distribusi stok pupuk subsidi petani dipercepat Pupuk Indonesia.",
            source="detik",
            published_at=datetime(2026, 11, 19, 5, 30, tzinfo=timezone.utc),
        ),
        _make_article(
            "b1",
            "Vale Operasikan Pabrik Nikel HPAL Baru di Sulawesi",
            "Vale Indonesia mengoperasikan pabrik nikel HPAL baru berkapasitas besar.",
            source="cnbc",
            published_at=datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc),
        ),
    ]

    # Act Batch 1
    res1 = clusterer.cluster_articles(
        batch1,
        active_topics=[],
        batch_time=datetime(2026, 11, 19, 7, 0, tzinfo=timezone.utc),
    )
    assert len(res1.new_topics) == 1
    assert res1.unmatched_article_keys == ["b1"]

    # Simulate MongoDB reload + fresh vocabulary state across batches
    reloaded_clusterer = TopicClusterer(
        similarity_threshold=0.60,
        min_cluster_size=2,
        title_weight=3,
        vectorizer=IncrementalTfidfVectorizer.from_state(res1.vocabulary_state),
    )

    # Batch 2 introduces brand-new words ("kaltim", "smelter") plus matching articles
    # and carries forward unmatched singleton b1 from DB (`topic_id is None`).
    batch2 = [
        batch1[2],  # unmatched singleton b1 still has topic_id=None in DB
        _make_article(
            "a3",
            "Pupuk Indonesia Tambah Stok Pupuk Subsidi Petani Kaltim",
            "Pupuk Indonesia memperluas stok pupuk subsidi petani hingga Kaltim.",
            source="republika",
            published_at=datetime(2026, 11, 19, 9, 0, tzinfo=timezone.utc),
        ),
        _make_article(
            "b2",
            "Pabrik Nikel HPAL Baru Mulai Dioperasikan Vale",
            "Smelter dan pabrik nikel HPAL baru mulai dioperasikan Vale Indonesia.",
            source="kompas",
            published_at=datetime(2026, 11, 19, 9, 30, tzinfo=timezone.utc),
        ),
    ]

    # Act Batch 2
    res2 = reloaded_clusterer.cluster_articles(
        batch2,
        active_topics=res1.new_topics,
        batch_time=datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc),
    )

    # Assert
    assert res2.article_topic_assignments["a3"] == "topik_20261119_001"
    assert len(res2.updated_topics) == 1
    assert res2.updated_topics[0]["article_count"] == 3
    assert len(res2.new_topics) == 1
    promoted_topic = res2.new_topics[0]
    assert promoted_topic["topic_id"] == "topik_20261119_002"
    assert promoted_topic["article_count"] == 2
    assert res2.article_topic_assignments["b1"] == "topik_20261119_002"
    assert res2.article_topic_assignments["b2"] == "topik_20261119_002"
    assert res2.unmatched_article_keys == []


def test_order_invariance_and_reproducibility():
    # Arrange
    articles = [
        _make_article(
            "k1",
            "Bos IMF Ungkap Efek AI ke Ekonomi Global",
            "IMF menyoroti dampak kecerdasan buatan terhadap pertumbuhan ekonomi global.",
            published_at=datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc),
        ),
        _make_article(
            "k2",
            "Dampak AI pada Ekonomi Global Disorot Bos IMF",
            "Pertumbuhan ekonomi global mulai dipengaruhi teknologi kecerdasan buatan menurut IMF.",
            published_at=datetime(2026, 11, 19, 5, 10, tzinfo=timezone.utc),
        ),
        _make_article(
            "k3",
            "Kementan Dorong Kemitraan Sawit Nasional",
            "Program kemitraan sawit meningkatkan produktivitas perkebunan rakyat nasional.",
            published_at=datetime(2026, 11, 19, 5, 20, tzinfo=timezone.utc),
        ),
        _make_article(
            "k4",
            "Produktivitas Kemitraan Sawit Nasional Didorong Kementan",
            "Kementerian Pertanian mendorong kemitraan sawit untuk mendongkrak produktivitas.",
            published_at=datetime(2026, 11, 19, 5, 30, tzinfo=timezone.utc),
        ),
    ]

    c_fwd = TopicClusterer(similarity_threshold=0.60, min_cluster_size=2)
    c_rev = TopicClusterer(similarity_threshold=0.60, min_cluster_size=2)
    ts = datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc)

    # Act
    res_fwd = c_fwd.cluster_articles(articles, batch_time=ts)
    res_rev = c_rev.cluster_articles(list(reversed(articles)), batch_time=ts)

    # Assert
    assert res_fwd.article_topic_assignments == res_rev.article_topic_assignments
    assert len(res_fwd.new_topics) == len(res_rev.new_topics) == 2
    for t_fwd, t_rev in zip(res_fwd.new_topics, res_rev.new_topics):
        assert t_fwd["topic_id"] == t_rev["topic_id"]
        assert t_fwd["label"] == t_rev["label"]
        assert t_fwd["keywords"] == t_rev["keywords"]
        sim = float(
            cosine_similarity(
                np.array(t_fwd["centroid_vector"]).reshape(1, -1),
                np.array(t_rev["centroid_vector"]).reshape(1, -1),
            )[0][0]
        )
        assert math.isclose(sim, 1.0, rel_tol=1e-9)


def test_topic_document_matches_contract_diagram_schema():
    # Arrange
    clusterer = TopicClusterer(similarity_threshold=0.60, min_cluster_size=2)
    articles = [
        _make_article(
            "s1",
            "Bank Indonesia Tahan Suku Bunga Acuan",
            "Rapat Dewan Gubernur Bank Indonesia mempertahankan suku bunga acuan demi stabilitas rupiah.",
        ),
        _make_article(
            "s2",
            "Suku Bunga Acuan Tetap Ditahan Bank Indonesia",
            "Bank Indonesia menjaga suku bunga acuan tetap stabil guna meredam gejolak nilai tukar rupiah.",
        ),
    ]

    # Act
    result = clusterer.cluster_articles(
        articles, batch_time=datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
    )

    # Assert
    assert len(result.new_topics) == 1
    topic = result.new_topics[0]
    expected_keys = {
        "topic_id",
        "label",
        "keywords",
        "centroid_vector",
        "status",
        "article_count",
        "first_seen_at",
        "last_updated_at",
    }
    assert set(topic.keys()) == expected_keys
    assert isinstance(topic["topic_id"], str) and topic["topic_id"].startswith("topik_20261122_")
    assert isinstance(topic["label"], str) and len(topic["label"]) > 0
    assert isinstance(topic["keywords"], list) and all(isinstance(k, str) for k in topic["keywords"])
    assert isinstance(topic["centroid_vector"], list) and len(topic["centroid_vector"]) > 0
    assert all(type(v) is float and math.isfinite(v) for v in topic["centroid_vector"])
    assert topic["status"] == "active"
    assert type(topic["article_count"]) is int and topic["article_count"] == 2
    assert isinstance(topic["first_seen_at"], datetime)
    assert isinstance(topic["last_updated_at"], datetime)
