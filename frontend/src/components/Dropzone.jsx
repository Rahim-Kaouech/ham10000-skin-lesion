import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function Dropzone({ onFile, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);

  const handle = (file) => {
    if (!file) return;
    if (!ACCEPTED.includes(file.type)) {
      setError("Format non supporté (JPEG, PNG, WebP ou BMP).");
      return;
    }
    setError(null);
    onFile(file);
  };

  return (
    <div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handle(e.dataTransfer.files?.[0]);
        }}
        className={`flex w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 text-center transition
          ${dragging ? "border-teal-500 bg-teal-50" : "border-slate-300 hover:border-teal-400 hover:bg-slate-50"}
          ${disabled ? "cursor-wait opacity-60" : "cursor-pointer"}`}
      >
        <UploadCloud className={`h-10 w-10 ${dragging ? "text-teal-600" : "text-slate-400"}`} />
        <div>
          <p className="font-medium text-slate-700">Glissez-déposez une image dermatoscopique</p>
          <p className="text-sm text-slate-500">ou cliquez pour parcourir · JPEG, PNG · 10 Mo max</p>
        </div>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        className="hidden"
        onChange={(e) => {
          handle(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}
