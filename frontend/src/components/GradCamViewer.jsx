import { useState } from "react";

const png = (b64) => `data:image/png;base64,${b64}`;

export default function GradCamViewer({ result }) {
  const [opacity, setOpacity] = useState(0.6);
  const [mode, setMode] = useState("overlay"); // overlay | side

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold whitespace-nowrap text-slate-700">Grad-CAM</h3>
        <div className="flex rounded-lg bg-slate-100 p-0.5 text-xs">
          {[["overlay", "Superposé"], ["side", "Comparer"]].map(([k, label]) => (
            <button key={k} type="button" onClick={() => setMode(k)}
              className={`rounded-md px-2.5 py-1 whitespace-nowrap ${mode === k ? "bg-white font-medium shadow-sm" : "text-slate-500"}`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {mode === "overlay" ? (
        <div className="relative mx-auto aspect-square w-full max-w-sm overflow-hidden rounded-xl bg-slate-900">
          <img src={png(result.image_png_base64)} alt="Image analysée" className="absolute inset-0 h-full w-full" />
          <img src={png(result.heatmap_png_base64)} alt="Carte Grad-CAM" style={{ opacity }}
            className="absolute inset-0 h-full w-full transition-opacity" />
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <img src={png(result.image_png_base64)} alt="Image analysée" className="aspect-square w-full rounded-xl" />
          <img src={png(result.overlay_png_base64)} alt="Superposition Grad-CAM" className="aspect-square w-full rounded-xl" />
        </div>
      )}

      {mode === "overlay" && (
        <label className="flex items-center gap-3 text-xs text-slate-500">
          Opacité
          <input type="range" min="0" max="1" step="0.05" value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))} className="flex-1 accent-teal-700" />
          <span className="w-8 font-mono">{Math.round(opacity * 100)}%</span>
        </label>
      )}
      <div className="flex items-center gap-2 text-[11px] text-slate-500">
        <span>Faible</span>
        <div className="h-2 flex-1 rounded-full" style={{ background: "linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f00)" }} />
        <span>Forte contribution à « {result.prediction.code} »</span>
      </div>
    </div>
  );
}
