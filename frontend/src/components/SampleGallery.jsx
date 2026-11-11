import { useEffect, useState } from "react";

// Images exportées par `python -m src.data.export_samples` (jeu de TEST uniquement)
export default function SampleGallery({ onFile, disabled }) {
  const [samples, setSamples] = useState(null);

  useEffect(() => {
    fetch("/samples/manifest.json")
      .then((r) => (r.ok ? r.json() : []))
      .then(setSamples)
      .catch(() => setSamples([]));
  }, []);

  const pick = async (s) => {
    const blob = await (await fetch(`/samples/${s.file}`)).blob();
    onFile(new File([blob], s.file, { type: blob.type || "image/jpeg" }), s.label);
  };

  if (samples === null) return <p className="text-sm text-slate-500">Chargement…</p>;
  if (samples.length === 0)
    return (
      <div className="rounded-xl bg-amber-50 p-4 text-sm text-amber-800">
        Aucune image d&apos;exemple. Lancez <code className="font-mono">python -m src.data.export_samples</code>{" "}
        après la préparation des données.
      </div>
    );

  return (
    <div>
    <p className="mb-2 text-xs text-slate-500">Images du jeu de test · le libellé est la vérité terrain</p>
    <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-7">
      {samples.map((s) => (
        <button
          key={s.file}
          type="button"
          disabled={disabled}
          onClick={() => pick(s)}
          title={s.label_name}
          className="group overflow-hidden rounded-xl border border-slate-200 text-left transition hover:-translate-y-0.5 hover:border-teal-400 hover:shadow-md disabled:opacity-50"
        >
          <img src={`/samples/${s.file}`} alt={s.label_name} className="aspect-square w-full object-cover" loading="lazy" />
          <div className="px-2 py-1 text-center font-mono text-xs font-semibold uppercase text-slate-700">
            {s.label}
          </div>
        </button>
      ))}
    </div>
    </div>
  );
}
