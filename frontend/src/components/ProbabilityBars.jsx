import { useEffect, useState } from "react";

export default function ProbabilityBars({ probabilities, groundTruth }) {
  // Démarre à 0 puis anime vers la valeur (transition CSS)
  const [shown, setShown] = useState(false);
  useEffect(() => {
    setShown(false);
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, [probabilities]);

  return (
    <ul className="space-y-2.5">
      {probabilities.map((p, i) => {
        const pct = p.probability * 100;
        const top = i === 0;
        const color = p.malignant
          ? top ? "bg-rose-600" : "bg-rose-300"
          : top ? "bg-teal-600" : "bg-teal-300";
        return (
          <li key={p.code}>
            <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
              <span className={`truncate ${top ? "font-semibold text-slate-900" : "text-slate-600"}`}>
                <span className="mr-1.5 font-mono text-xs uppercase text-slate-400">{p.code}</span>
                {p.name}
                {p.malignant && <span className="ml-1.5 rounded bg-rose-50 px-1 text-[10px] font-semibold text-rose-700">malin</span>}
                {groundTruth === p.code && <span className="ml-1.5 rounded bg-indigo-50 px-1 text-[10px] font-semibold text-indigo-700">vérité terrain</span>}
              </span>
              <span className="font-mono tabular-nums text-slate-700">{pct.toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full ${color} transition-[width] duration-700 ease-out`}
                style={{ width: shown ? `${Math.max(pct, 0.5)}%` : "0%", transitionDelay: `${i * 60}ms` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
