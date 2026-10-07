import time

import pytest

from conftest import requires_cuda


@requires_cuda
@pytest.mark.slow
def test_server_job_roundtrip():
    from fastapi.testclient import TestClient

    from sculpture.server.app import app

    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    d = c.get("/api/defaults").json()
    assert len(d["cameras"]) == 2
    models = c.get("/api/models", params={"dir": "assets/models"}).json()
    assert len(models) >= 3
    assert c.get(models[0]["thumbnail_url"]).headers["content-type"] == "image/png"
    assert c.get(models[0]["mesh_url"]).content[:4] == b"glTF"
    targets = c.get("/api/targets").json()
    assert any(t["name"] == "view_0.png" for t in targets)

    req = {
        "models_dir": "assets/models",
        "targets": ["assets/targets/view_0.png", "assets/targets/view_1.png"],
        "cameras": d["cameras"],
        "methods": ["beam"],
        "preset": "fast",
        "overrides": {"beam": {"max_runtime_s": 8, "final_refine_steps": 10}},
    }
    job_id = c.post("/api/jobs", json=req).json()["job_id"]
    with c.websocket_connect(f"/api/jobs/{job_id}/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "status"
    t0 = time.time()
    while time.time() - t0 < 180:
        st = c.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(1)
    assert st["status"] == "done", st.get("error")
    assert st["comparison"]["best_method"] == "beam"
    beam = st["methods"]["beam"]
    assert beam["metrics"]["min_view_iou"] > 0.3
    assert beam["assembly"]["objects"]
    assert c.get(st["preprocessing"]["hull_url"]).status_code == 200
    assert c.get(beam["result_dir_url"] + "/assembly.glb").status_code == 200
