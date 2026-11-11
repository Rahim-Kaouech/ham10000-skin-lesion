import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Camera, Images, Info, Loader2, ShieldCheck, Upload } from "lucide-react";
import { api } from "./services/api.js";
import Dropzone from "./components/Dropzone.jsx";
import SampleGallery from "./components/SampleGallery.jsx";
import WebcamCapture from "./components/WebcamCapture.jsx";
import ProbabilityBars from "./components/ProbabilityBars.jsx";
import GradCamViewer from "./components/GradCamViewer.jsx";
import LatencyPanel from "./components/LatencyPanel.jsx";
import ModelInfo from "./components/ModelInfo.jsx";

const TABS = [
  { id: "upload", label: "Importer", icon: Upload },
  { id: "gallery", label: "Exemples", icon: Images },
  { id: "webcam", label: "Webcam", icon: Camera },
];

function HealthBadge({ health }) {
  if (!health) return <span className="text-xs text-slate-400">connexion…</span>;
  const ok = health.status === "ok";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${ok ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-amber-500"}`} />
      {ok ? "Modèle prêt" : health.status === "offline" ? "API hors ligne" : "Modèle non chargé"}
      {health.gpu_available && <span className="ml-1 rounded bg-emerald-100 px-1">GPU</span>}
    </span>
  );
}

export default function App() {
  const [tab, setTab] = useState("upload");
  const [health, setHealth] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [result, setResult] = useState(null);
  const [clientMs, setClientMs] = useState(0);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);

  useEffect(() => {
    const refresh = () =>
      api.health()
        .then(({ data }) => setHealth(data))
        .catch(() => setHealth({ status: "offline" }));
    refresh();
    api.metadata().then(({ data }) => setMetadata(data)).catch(() => {});
    const id = setInterval(refresh, 15000);
    return () => clearInterval(id);
  }, []);

  const analyze = useCallback(async (file, opts = {}) => {
    setBusy(true);
    setError(null);
    if (!opts.live) {
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(file);
      });
      setGroundTruth(opts.label || null);
    }
    try {
      const { data, clientMs: ms } = await api.visualize(file);
      setResult(data);
      setClientMs(ms);
      setHistory((h) => [...h.slice(-29), ms]);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-teal-700">
              <div className="h-3.5 w-3.5 rounded-full bg-amber-400" />
            </div>
            <div>
              <h1 className="text-lg leading-tight font-semibold text-slate-900">DermaLens</h1>
              <p className="text-xs text-slate-500">Classification de lésions pigmentées · HAM10000 · Grad-CAM</p>
            </div>
          </div>
          <HealthBadge health={health} />
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        {/* ---------------- Entrée */}
        <section className="space-y-6">
          <div className="card p-5">
            <div className="mb-4 flex gap-1 rounded-xl bg-slate-100 p-1">
              {TABS.map(({ id, label, icon: Icon }) => (
                <button key={id} type="button" onClick={() => setTab(id)}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-lg py-2 text-sm transition ${tab === id ? "bg-white font-medium text-teal-800 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
                  <Icon className="h-4 w-4" /> {label}
                </button>
              ))}
            </div>
            {tab === "upload" && <Dropzone onFile={(f) => analyze(f)} disabled={busy} />}
            {tab === "gallery" && <SampleGallery onFile={(f, label) => analyze(f, { label })} disabled={busy} />}
            {tab === "webcam" && <WebcamCapture onFrame={analyze} busy={busy} />}

            {preview && tab !== "webcam" && (
              <div className="mt-4 flex items-center gap-3 rounded-xl bg-slate-50 p-2">
                <img src={preview} alt="Aperçu" className="h-16 w-16 rounded-lg object-cover" />
                <span className="text-sm text-slate-600">
                  {busy ? "Analyse en cours…" : "Dernière image analysée"}
                </span>
                {busy && <Loader2 className="ml-auto h-5 w-5 animate-spin text-teal-700" />}
              </div>
            )}
          </div>

          <div className="card p-5">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-700">
              <Info className="h-4 w-4" /> Modèle servi
            </h2>
            {metadata ? <ModelInfo metadata={metadata} /> : <p className="text-sm text-slate-400">Indisponible</p>}
          </div>

          <p className="flex gap-2 text-xs leading-relaxed text-slate-500">
            <ShieldCheck className="h-4 w-4 shrink-0" />
            Outil pédagogique d&apos;aide à l&apos;analyse. Il ne remplace en aucun cas l&apos;avis d&apos;un
            dermatologue ni un examen histopathologique.
          </p>
        </section>

        {/* ---------------- Résultats */}
        <section className="space-y-6">
          {error && (
            <div className="card flex items-start gap-3 border-red-200 bg-red-50 p-4 text-sm text-red-700">
              <AlertTriangle className="h-5 w-5 shrink-0" /> {error}
            </div>
          )}

          {!result && !error && (
            <div className="card flex h-full min-h-80 flex-col items-center justify-center p-10 text-center text-slate-400">
              <Images className="mb-3 h-10 w-10" />
              <p>Importez une image, choisissez un exemple ou utilisez la webcam.</p>
            </div>
          )}

          {result && (
            <>
              <div className={`card overflow-hidden ${busy ? "opacity-80" : ""}`}>
                <div className={`flex items-center gap-4 px-5 py-4 ${result.risk_alert ? "bg-rose-50" : "bg-teal-50"}`}>
                  {result.risk_alert ? (
                    <AlertTriangle className="h-8 w-8 shrink-0 text-rose-600" />
                  ) : (
                    <ShieldCheck className="h-8 w-8 shrink-0 text-teal-700" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className={`text-xs font-semibold uppercase tracking-wide ${result.risk_alert ? "text-rose-700" : "text-teal-700"}`}>
                      {result.risk_alert ? "Lésion potentiellement maligne · avis médical recommandé" : "Lésion probablement bénigne"}
                    </div>
                    <div className="truncate text-xl font-semibold text-slate-900">{result.prediction.name}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-2xl font-bold tabular-nums text-slate-900">
                      {(result.prediction.probability * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-500">confiance</div>
                  </div>
                </div>
                {groundTruth && (
                  <div className="border-t border-slate-100 px-5 py-2 text-xs text-slate-600">
                    Vérité terrain : <span className="font-mono font-semibold uppercase">{groundTruth}</span>{" "}
                    {groundTruth === result.prediction.code ? "✓ correcte" : "✗ erreur du modèle"}
                  </div>
                )}
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div className="card p-5">
                  <h3 className="mb-4 text-sm font-semibold text-slate-700">Probabilités par classe</h3>
                  <ProbabilityBars probabilities={result.probabilities} groundTruth={groundTruth} />
                </div>
                <div className="card p-5">
                  <GradCamViewer result={result} />
                </div>
              </div>

              <div className="card p-5">
                <LatencyPanel result={result} clientMs={clientMs} history={history} />
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
