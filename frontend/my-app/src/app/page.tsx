"use client";

import { useState, useRef, useEffect } from "react";
import NavigationBar from "@/components/navbar";
import { Montserrat } from "next/font/google";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { motion, AnimatePresence } from "framer-motion";

const montserratFont = Montserrat({
  subsets: ["latin"],
  weight: "400",
});

export default function Home() {
  const [value, setValue] = useState("");
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const outputRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  // --- COMPONENT: RESULT CARD ---
  function ResultCard({ r }: { r: any }) {
    const [open, setOpen] = useState(false);
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
      await navigator.clipboard.writeText(r.text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    };

    return (
      <div className="border rounded-lg p-4 bg-white shadow-sm mt-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold">{r.recipe_name}</h3>
            <p className="text-xs text-gray-500">Score: {r.score?.toFixed(3)}</p>
          </div>

          <div className="flex gap-3">
            <button 
              className="text-xs text-indigo-600 font-medium hover:underline" 
              onClick={() => setOpen(!open)}
            >
              {open ? "Tutup" : "Baca Detail"}
            </button>
            <button
              className="text-xs border px-2 py-1 rounded hover:bg-gray-50 transition"
              onClick={handleCopy}
            >
              {copied ? "✓ Disalin" : "Salin"}
            </button>
          </div>
        </div>

        {open && (
          <div className="mt-3 pt-3 border-t border-gray-100">
            <div className="prose prose-sm prose-slate max-w-none 
                          prose-headings:font-semibold prose-headings:text-gray-800
                          prose-h3:text-base prose-h3:mt-4 prose-h3:mb-2
                          prose-p:my-2 prose-p:text-gray-700
                          prose-ul:my-2 prose-ul:ml-4
                          prose-ol:my-2 prose-ol:ml-4
                          prose-li:my-1">
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                {r.text}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Auto expand textarea
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [value]);

  // Auto scroll to top when new response
  useEffect(() => {
    if (outputRef.current && (response || loading)) {
      outputRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [loading, response]);

  const handleSubmit = async () => {
    if (!value.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      console.log("🚀 FRONTEND - Sending request:", value);
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value, top_k: 3 }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      console.log("📦 FRONTEND - Received response:", data);

      if (data.results?.[0]?.error) {
        setError(data.results[0].error);
      } else {
        setResponse(data);
      }
    } catch (err: any) {
      console.error("❌ FRONTEND - Fetch error:", err);
      setError(err?.message || "Error saat mengambil data dari server.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className={`bg-white min-h-screen ${montserratFont.className}`}>
      <NavigationBar />

      {/* Main Container */}
      <main className="pt-24 pb-32 px-4 md:px-8 lg:px-40">
        <div
          ref={outputRef}
          className="p-4 min-h-[200px] max-h-[calc(100vh-200px)] overflow-y-auto space-y-8"
        >
          {loading ? (
            <div className="flex flex-col items-center justify-center space-y-4 py-10 opacity-60">
              <div className="w-8 h-8 border-4 border-gray-300 border-t-blue-600 rounded-full animate-spin"></div>
              <p className="text-gray-600">Sedang meracik resep...</p>
            </div>
          ) : error ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="p-4 bg-red-50 text-red-600 rounded-lg border border-red-100"
            >
              <strong>Error:</strong> {error}
            </motion.div>
          ) : response ? (
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                {/* 1. MAIN ANSWER (Proper Markdown Styling) */}
                <div className="bg-white rounded-2xl">
                  <div className="prose prose-slate prose-lg max-w-none
                                prose-headings:font-bold prose-headings:text-gray-900
                                prose-h2:text-2xl prose-h2:mt-6 prose-h2:mb-4 prose-h2:border-b prose-h2:pb-2 prose-h2:border-gray-200
                                prose-h3:text-xl prose-h3:mt-5 prose-h3:mb-3 prose-h3:text-gray-800
                                prose-p:text-gray-700 prose-p:leading-relaxed prose-p:my-3
                                prose-ul:my-3 prose-ul:ml-6 prose-ul:space-y-2
                                prose-ol:my-3 prose-ol:ml-6 prose-ol:space-y-2
                                prose-li:text-gray-700 prose-li:leading-relaxed
                                prose-strong:text-gray-900 prose-strong:font-semibold
                                prose-code:bg-gray-100 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm
                                prose-pre:bg-gray-900 prose-pre:text-gray-100">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]} 
                      rehypePlugins={[rehypeRaw]}
                    >
                      {response.answer}
                    </ReactMarkdown>
                  </div>
                </div>

                <hr className="my-8 border-gray-200" />

                {/* 2. REFERENCE LIST */}
                {response.results && response.results.length > 1 && (
                  <div className="space-y-4">
                    <h4 className="font-semibold text-gray-400 text-sm uppercase tracking-wider mb-4">
                      Referensi Lainnya
                    </h4>
                    {response.results.slice(1).map((r: any, i: number) => (
                      <ResultCard key={i} r={r} />
                    ))}
                  </div>
                )}

              </motion.div>
            </AnimatePresence>
          ) : (
            <div className="text-center text-gray-400 py-20">
              <p className="text-xl font-light">Ketik bahan atau nama resep untuk memulai...</p>
              <p className="text-sm mt-2">Contoh: "apple pie", "roti gandum", "kue coklat"</p>
            </div>
          )}
        </div>
      </main>

      {/* Input Area */}
      <div className="fixed bottom-0 w-full bg-white/95 backdrop-blur-sm pb-8 pt-4 px-4 md:px-8 lg:px-40 border-t border-gray-100">
        <div className="relative">
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Apa yang ingin Anda cari hari ini? (contoh: resep apple pie, kue coklat, dll)"
            disabled={loading}
            className="w-full p-4 border border-gray-300 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50 shadow-sm pr-12 disabled:bg-gray-50 disabled:cursor-not-allowed"
            rows={1}
          />
          
          <button 
            onClick={handleSubmit}
            disabled={loading || !value.trim()}
            className="absolute right-3 top-3 p-2 bg-blue-100 hover:bg-blue-200 rounded-xl text-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "⏳" : "➤"}
          </button>
        </div>
        
        <p className="text-center text-xs text-gray-400 mt-2">
          ⚠️ Hasil berdasarkan data resep yang tersedia di database
        </p>
      </div>
    </div>
  );
}