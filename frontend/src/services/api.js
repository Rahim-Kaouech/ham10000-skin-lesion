// Client HTTP vers l'API FastAPI. En Docker, nginx proxifie /api -> backend:8000.
const API_URL = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Erreur HTTP ${status}`);
    this.status = status;
  }
}

async function request(path, options = {}) {
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new ApiError(0, "API injoignable : le backend est-il démarré ?");
  }
  const clientMs = performance.now() - t0;
  if (!res.ok) {
    let detail;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      detail = res.statusText;
    }
    throw new ApiError(res.status, detail);
  }
  return { data: await res.json(), clientMs };
}

function formData(file, filename = "image.jpg") {
  const fd = new FormData();
  fd.append("file", file, file.name || filename);
  return fd;
}

export const api = {
  health: () => request("/health"),
  metadata: () => request("/model/metadata"),
  stats: () => request("/monitoring/stats"),
  predict: (file) => request("/predict", { method: "POST", body: formData(file) }),
  // Prédiction + Grad-CAM en un seul appel (JSON avec images base64)
  visualize: (file, alpha = 0.45) =>
    request(`/predict/visualize?format=json&alpha=${alpha}`, {
      method: "POST",
      body: formData(file),
    }),
};
