type DeveloperModePanelProps = {
  enabled: boolean
}

const samplePayload = {
  run_id: '4af96145-01e0-4210-a487-87f41512ae6f',
  retrieval_backend: 'chroma',
  top_k: 5,
  model: 'xgboost + llama-3.1-8b-instant',
  latency_ms: {
    retrieval: 3401.09,
    ml_prediction: 197.45,
    rag_generation: 1536.9,
  },
}

export function DeveloperModePanel({ enabled }: DeveloperModePanelProps) {
  if (!enabled) {
    return null
  }
  return (
    <section className="glass-card border-cyan-500/30 p-5">
      <p className="section-title text-cyan-300">Developer Mode</p>
      <pre className="mt-3 overflow-x-auto rounded-xl bg-slate-950/80 p-3 text-xs text-cyan-100">
        {JSON.stringify(samplePayload, null, 2)}
      </pre>
    </section>
  )
}
