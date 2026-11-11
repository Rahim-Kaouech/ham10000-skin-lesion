import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Pause, Play } from "lucide-react";

// Capture une image carrée centrée (le modèle fait un center-crop)
function grabFrame(video, size = 512) {
  const canvas = document.createElement("canvas");
  const side = Math.min(video.videoWidth, video.videoHeight);
  canvas.width = canvas.height = size;
  canvas
    .getContext("2d")
    .drawImage(video, (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side, 0, 0, size, size);
  return new Promise((resolve) =>
    canvas.toBlob((b) => resolve(new File([b], "webcam.jpg", { type: "image/jpeg" })), "image/jpeg", 0.9),
  );
}

export default function WebcamCapture({ onFrame, busy }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);
  const [live, setLive] = useState(false);
  const [error, setError] = useState(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setActive(false);
    setLive(false);
  }, []);

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setActive(true);
      setError(null);
    } catch (e) {
      setError(`Webcam indisponible : ${e.message}`);
    }
  };

  useEffect(() => stop, [stop]);

  // Mode continu : une nouvelle requête dès que la précédente est terminée
  useEffect(() => {
    if (!live || busy || !active) return undefined;
    const id = setTimeout(async () => {
      if (videoRef.current?.videoWidth) onFrame(await grabFrame(videoRef.current), { live: true });
    }, 250);
    return () => clearTimeout(id);
  }, [live, busy, active, onFrame]);

  return (
    <div className="space-y-3">
      <div className="relative aspect-video overflow-hidden rounded-xl bg-slate-900">
        <video ref={videoRef} playsInline muted className="h-full w-full object-cover" />
        {active && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className="aspect-square h-[85%] rounded-lg border-2 border-dashed border-white/70" />
          </div>
        )}
        {!active && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-400">
            Caméra inactive
          </div>
        )}
        {live && (
          <span className="absolute top-3 left-3 flex items-center gap-1.5 rounded-full bg-red-600 px-2.5 py-1 text-xs font-semibold text-white">
            <span className="h-2 w-2 animate-pulse rounded-full bg-white" /> LIVE
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {!active ? (
          <button type="button" onClick={start} className="inline-flex items-center gap-2 rounded-lg bg-teal-700 px-4 py-2 text-sm font-medium text-white hover:bg-teal-800">
            <Camera className="h-4 w-4" /> Activer la caméra
          </button>
        ) : (
          <>
            <button
              type="button"
              disabled={busy || live}
              onClick={async () => onFrame(await grabFrame(videoRef.current), { live: false })}
              className="inline-flex items-center gap-2 rounded-lg bg-teal-700 px-4 py-2 text-sm font-medium text-white hover:bg-teal-800 disabled:opacity-50"
            >
              <Camera className="h-4 w-4" /> Capturer et analyser
            </button>
            <button
              type="button"
              onClick={() => setLive((v) => !v)}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium ${live ? "bg-red-100 text-red-700 hover:bg-red-200" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`}
            >
              {live ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {live ? "Arrêter le flux" : "Inférence continue"}
            </button>
            <button type="button" onClick={stop} className="inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm text-slate-500 hover:bg-slate-100">
              <CameraOff className="h-4 w-4" /> Couper
            </button>
          </>
        )}
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
