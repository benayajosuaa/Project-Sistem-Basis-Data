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

  // 🪶 Auto resize textarea
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
    el.style.overflowY = el.scrollHeight > 140 ? "auto" : "hidden";
  }, [value]);

  // 🔄 Auto-scroll behaviour yang benar:
  useEffect(() => {
    if (!outputRef.current) return;

    if (loading) {
      // Saat loading, biar posisi ke bawah untuk indikator “mencari...”
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    } else {
      // Saat jawaban muncul, scroll ke atas biar user baca dari awal
      outputRef.current.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    }
  }, [loading, response]);

  // 🚀 Kirim pertanyaan ke backend
  const handleSubmit = async () => {
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

      if (!res.ok) throw new Error("Gagal mengambil data dari server");

      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      console.error(err);
      setError("❌ Terjadi kesalahan saat mengambil data dari server.");
    } finally {
      setLoading(false);
      setValue("");
    }
  };

  // ⌨️ Enter = kirim, Shift+Enter = baris baru
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className={`overflow-x-hidden bg-white min-h-screen ${montserratFont.className}`}>
      {/* Navbar */}
      <div className="fixed z-20 w-full">
        <NavigationBar />
      </div>

      {/* Main Content */}
      <main className="relative z-10 h-screen text-black pl-40 pr-40 pt-24 pb-32">
        <div className="flex flex-col gap-4">
          <div
            ref={outputRef}
            className="p-4 rounded-2xl min-h-[200px] max-h-[calc(100vh-280px)] overflow-y-auto space-y-6"
          >
            {loading ? (
              <p className="animate-pulse text-gray-600">
                ⏳ Sedang mencari jawaban dari database kami ...
              </p>
            ) : error ? (
              <p className="text-red-600">{error}</p>
            ) : response ? (
              <AnimatePresence>
                <motion.div
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="flex flex-col space-y-6"
                >
                  {/* 💬 Jawaban utama */}
                  <div className="flex justify-start">
                    <div className="px-5 py-4 rounded-2xl w-full ">
                      <div className="prose prose-slate max-w-none text-gray-800 leading-relaxed">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeRaw]}
                          components={{
                            a: (props) => (
                              <a
                                {...props}
                                className="text-blue-600 underline hover:text-blue-800"
                                target="_blank"
                                rel="noreferrer"
                              />
                            ),
                            ul: (props) => (
                              <ul {...props} className="list-disc pl-6 space-y-1" />
                            ),
                            ol: (props) => (
                              <ol {...props} className="list-decimal pl-6 space-y-1" />
                            ),
                            li: (props) => <li {...props} className="mb-1" />,
                            strong: (props) => (
                              <strong {...props} className="font-semibold text-gray-900" />
                            ),
                            h1: (props) => (
                              <h1 {...props} className="text-2xl font-bold mt-4 mb-2" />
                            ),
                            h2: (props) => (
                              <h2 {...props} className="text-xl font-semibold mt-3 mb-2" />
                            ),
                            h3: (props) => (
                              <h3 {...props} className="text-lg font-semibold mt-2 mb-1" />
                            ),
                            p: (props) => (
                              <p {...props} className="mb-2 whitespace-pre-line" />
                            ),
                            code: (props) => (
                              <code
                                {...props}
                                className="bg-gray-100 px-1 py-0.5 rounded text-sm text-gray-800"
                              />
                            ),
                          }}
                        >
                          {response.answer || "_Tidak ada jawaban ditemukan._"}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>

                  {/* 📚 Hasil pencarian tambahan */}
                  {response.results?.length > 0 && (
                    <div className="space-y-3 pl-2">
                      {response.results.map((r: any, i: number) => (
                        <div key={i} className="text-gray-700">
                          {r.recipe_name && (
                            <p className="font-semibold text-blue-700">
                              {r.recipe_name}
                            </p>
                          )}
                          <p className="text-sm text-gray-600 whitespace-pre-line">
                            {r.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            ) : (
              <p className="text-gray-400">
                Ketik pertanyaan pada kolom di bawah untuk mulai mencari resep dari database kami... <br/>
                sebisa mungkin akan dijawab selagi berhubungan dengan data yang dipunya ☝️🤓 <br/><br/>
                selamat mencoba ~
              </p>
            )}
          </div>
        </div>
      </main>

      {/* 📝 Input Section */}
      <div className="fixed z-20 bottom-0 flex flex-col gap-y-4 w-full pl-40 pr-40 pb-6 text-black bg-white/90 backdrop-blur-sm ">
        <form
          className="w-full"
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
        >
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              loading ? "Sedang mencari jawaban..." : "Apa yang mau anda cari hari ini ?"
            }
            disabled={loading}
            className={`top w-full bg-white p-3 border border-slate-400 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all overflow-y-auto ${
              loading ? "opacity-60 cursor-not-allowed" : ""
            }`}
            style={{ maxHeight: "130px" }}
          />
        </form>

        <div className="flex justify-center items-center text-xs text-gray-500">
          <p>
            ⚠️ jawaban yang muncul hanya berdasarkan database yang ada
          </p>
        </div>
      </div>
    </div>
  );
}
