import { Timer } from "lucide-react";

function Stat({ label, value, hint }) {
  return (
    <div className="rounded-xl bg-slate-50 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="font-mono text-lg font-semibold tabular-nums text-slate-800">
        {value.toFixed(1)}<span className="ml-0.5 text-xs font-normal text-slate-500">ms</span>
      </div>
      {hint && <div className="text-[11px] text-slate-400">{hint}</div>}
    </div>
  );
}

export default function LatencyPanel({ result, clientMs, history }) {
  const max = Math.max(...history, 1);
  return (
    <div className="space-y-3">
      <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-700">
        <Timer className="h-4 w-4" /> Latence
      </h3>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Client (aller-retour)" value={clientMs} hint="réseau + serveur" />
        <Stat label="Serveur total" value={result.server_total_ms} />
        <Stat label="Inférence ONNX" value={result.inference_ms} />
        <Stat label="Prétraitement" value={result.preprocessing_ms} />
      </div>
      {history.length > 1 && (
        <div>
          <div className="mb-1 text-[11px] text-slate-500">
            {history.length} dernières requêtes (client) · moyenne{" "}
            {(history.reduce((a, b) => a + b, 0) / history.length).toFixed(0)} ms
          </div>
          <div className="flex h-10 items-end gap-0.5">
            {history.map((v, i) => (
              <div key={i} className="flex-1 rounded-t bg-teal-500/70" style={{ height: `${(v / max) * 100}%` }} title={`${v.toFixed(0)} ms`} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
