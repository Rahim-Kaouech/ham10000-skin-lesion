const fmt = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—");

export default function ModelInfo({ metadata }) {
  if (!metadata) return null;
  const test = metadata.metrics?.test || {};
  return (
    <div className="space-y-3 text-sm">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        <dt className="text-slate-500">Architecture</dt>
        <dd className="font-mono text-slate-800">{metadata.architecture}</dd>
        <dt className="text-slate-500">Variante ONNX</dt>
        <dd className="font-mono text-slate-800">{metadata.variant}</dd>
        <dt className="text-slate-500">Entrée</dt>
        <dd className="font-mono text-slate-800">{metadata.input_size}×{metadata.input_size}</dd>
        <dt className="text-slate-500">Version</dt>
        <dd className="truncate font-mono text-slate-800">{metadata.version}</dd>
      </dl>
      <div className="grid grid-cols-3 gap-2">
        {[["Bal. acc.", test.balanced_accuracy], ["F1 macro", test.f1_macro], ["Rappel mel", test.recall_mel]].map(
          ([label, v]) => (
            <div key={label} className="rounded-lg bg-teal-50 px-2 py-1.5 text-center">
              <div className="font-mono font-semibold text-teal-800">{fmt(v)}</div>
              <div className="text-[11px] text-teal-700">{label} (test)</div>
            </div>
          ),
        )}
      </div>
    </div>
  );
}
