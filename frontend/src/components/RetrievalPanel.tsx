type RetrievalChunk = {
  id: string
  source: string
  score: number
  snippet: string
}

type RetrievalPanelProps = {
  chunks: RetrievalChunk[]
}

export function RetrievalPanel({ chunks }: RetrievalPanelProps) {
  return (
    <section className="glass-card p-5">
      <p className="section-title">Retrieval Evidence</p>
      <div className="mt-4 space-y-3">
        {chunks.map((chunk) => (
          <article key={chunk.id} className="rounded-xl border border-slate-700 bg-slate-900/60 p-3">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>{chunk.source}</span>
              <span className="rounded-full bg-indigo-500/20 px-2 py-0.5 text-indigo-200">
                similarity {chunk.score.toFixed(2)}
              </span>
            </div>
            <p className="mt-2 text-sm text-slate-200">{chunk.snippet}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
