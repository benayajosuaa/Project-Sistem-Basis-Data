"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import NavigationBar from "@/components/navbar";
import { Montserrat } from "next/font/google";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks"; // npm i remark-breaks
import rehypeRaw from "rehype-raw";
import { motion, AnimatePresence } from "framer-motion";

const montserratFont = Montserrat({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

/* -----------------------------
   TYPES
------------------------------ */
type ResultItem = {
  score?: number;
  text?: string;
  recipe_name?: string;
  error?: string;
};

type ApiResponse = {
  answer?: string;
  results?: ResultItem[];
};

export default function Home() {
  const [value, setValue] = useState("");
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const outputRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  /* -----------------------------
     HELPER: MARKDOWN RENDERER
  ------------------------------ */
  const MarkdownRenderer = ({ content }: { content: string }) => (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      rehypePlugins={[rehypeRaw]}
      components={{
        ul: ({ ...props }) => <ul className="list-disc list-outside ml-5 my-3 space-y-1 text-gray-700" {...props} />,
        ol: ({ ...props }) => <ol className="list-decimal list-outside ml-5 my-3 space-y-2 text-gray-700" {...props} />,
        li: ({ ...props }) => <li className="pl-1 leading-relaxed" {...props} />,
        h1: ({ ...props }) => <h1 className="text-2xl font-bold mt-6 mb-4 text-gray-900" {...props} />,
        h2: ({ ...props }) => <h2 className="text-xl font-bold mt-5 mb-3 text-gray-800 border-b pb-2" {...props} />,
        h3: ({ ...props }) => <h3 className="text-lg font-semibold mt-4 mb-2 text-gray-800" {...props} />,
        p: ({ ...props }) => <p className="my-3 leading-7 text-gray-700" {...props} />,
        strong: ({ ...props }) => <strong className="font-semibold text-gray-900" {...props} />,
      }}
    >
      {content}
    </ReactMarkdown>
  );

  /* -----------------------------
     NORMALIZE MARKDOWN
  ------------------------------ */
  const normalizeMarkdown = (md?: string) => {
    if (!md) return "";
    // Memastikan baris baru dihormati
    return md.replace(/\\n/g, "\n").trim();
  };

  /* -----------------------------
     RESULT CARD
  ------------------------------ */
  function ResultCard({ r }: { r: ResultItem }) {
    const [open, setOpen] = useState(false);
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
      try {
        await navigator.clipboard.writeText(r.text || "");
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      } catch {
        setCopied(false);
      }
    };

    return (
      <div className="border border-gray-200 rounded-lg p-5 bg-white shadow-sm mt-4 hover:shadow-md transition-shadow">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">{r.recipe_name || "Tanpa nama"}</h3>
            <p className="text-xs text-gray-500 mt-1">
              Sumber: dataset (Score: {typeof r.score === "number" ? r.score.toFixed(2) : "—"})
            </p>
          </div>

          <div className="flex gap-3">
            <button
              className="text-xs text-indigo-600 font-medium hover:underline"
              onClick={() => setOpen((s) => !s)}
            >
              {open ? "Tutup" : "Baca Sumber"}
            </button>

            <button
              className="text-xs border px-3 py-1 rounded hover:bg-gray-50 transition"
              onClick={handleCopy}
            >
              {copied ? "✓ Disalin" : "Salin"}
            </button>
          </div>
        </div>

        {open && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-4 pt-4 border-t border-gray-100"
          >
            <div className="text-sm">
              <MarkdownRenderer content={normalizeMarkdown(r.text || "")} />
            </div>
          </motion.div>
        )}
      </div>
    );
  }

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [value]);

  useEffect(() => {
    if (outputRef.current && !loading && response) {
      window.scrollTo({ top: 100, behavior: 'smooth' });
    }
  }, [loading, response]);

  const handleSubmit = useCallback(async () => {
    if (!value.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value, top_k: 3 }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error: ${res.status} ${text}`);
      }

      const data: ApiResponse = await res.json();

      if (data.results?.[0]?.error) {
        setError(data.results[0].error);
        return;
      }

      const safe: ApiResponse = {
        answer: normalizeMarkdown(data.answer),
        results: data.results?.map((r) => ({
          ...r,
          text: normalizeMarkdown(r.text),
        })),
      };

      setResponse(safe);
    } catch (err: any) {
      setError(err?.message || "Error mengambil data.");
    } finally {
      setLoading(false);
    }
  }, [API_URL, value, loading]);

  return (
    <div className={`bg-white min-h-screen ${montserratFont.className} text-gray-800`}>
      <NavigationBar />

      <main className="pt-24 pb-40 px-6 md:px-20 lg:px-40 max-w-5xl mx-auto">
        <div ref={outputRef} className="p-4 min-h-[200px] space-y-8">

          {/* LOADING */}
          {loading ? (
            <div className="flex flex-col items-center justify-center space-y-4 py-20 opacity-60">
              <div className="w-10 h-10 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>
              <p className="font-medium text-gray-500">Sedang meracik resep...</p>
            </div>
          ) : null}

          {/* ERROR */}
          {error ? (
            <div className="p-4 bg-red-50 text-red-600 rounded-lg border border-red-100 flex items-center gap-2">
              <span>⚠️</span> {error}
            </div>
          ) : null}

          {/* HAS RESPONSE */}
          {!loading && !error && response ? (
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                <div className="bg-white">
                  <MarkdownRenderer content={response.answer || ""} />
                </div>

                <hr className="my-10 border-gray-100" />

                <div className="space-y-4">
                  <h4 className="font-semibold text-gray-400 text-xs uppercase tracking-widest mb-6">
                    Referensi Dataset
                  </h4>

                  {response.results?.map((r, i) => (
                    <ResultCard key={i} r={r} />
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>
          ) : null}

          {/* EMPTY STATE */}
          {!loading && !error && !response ? (
            <div className="text-center text-gray-300 py-32 select-none">
              <p className="text-6xl mb-4 grayscale opacity-50">👨‍🍳</p>
              <p className="text-xl font-light">Ketik bahan atau nama kue untuk memulai...</p>
            </div>
          ) : null}
        </div>
      </main>

      {/* INPUT BAR */}
      <div className="fixed bottom-0 w-full bg-white/90 backdrop-blur-md pb-8 pt-6 px-8 md:px-40 border-t border-gray-100 z-50">
        <div className="relative max-w-4xl mx-auto">
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void handleSubmit();
              }
            }}
            placeholder="Apa yang ingin Anda cari hari ini?"
            disabled={loading}
            className="w-full p-4 pl-6 border border-gray-200 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 shadow-lg shadow-gray-100 pr-14 text-gray-700 placeholder:text-gray-400"
            rows={1}
            style={{ minHeight: "60px" }}
          />

          <button
            onClick={() => void handleSubmit()}
            disabled={loading || !value.trim()}
            className="absolute right-3 top-3 p-2.5 bg-gray-100 hover:bg-blue-600 hover:text-white rounded-xl text-gray-500 transition-all disabled:opacity-50 disabled:hover:bg-gray-100 disabled:hover:text-gray-500"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          </button>
        </div>

        <p className="text-center text-[10px] text-gray-400 mt-3">
          ⚠️ Jawaban hanya berdasarkan data dataset.
        </p>
      </div>
    </div>
  );
}