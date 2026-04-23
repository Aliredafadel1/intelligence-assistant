import { useState } from 'react'
import { ChatAssistant } from './components/ChatAssistant'
import { DeveloperModePanel } from './components/DeveloperModePanel'
import { Header } from './components/Header'
import { InputPanel } from './components/InputPanel'
import { InsightsCharts } from './components/InsightsCharts'
import { PipelineFlow } from './components/PipelineFlow'
import { RecentQueriesTable } from './components/RecentQueriesTable'
import { RetrievalPanel } from './components/RetrievalPanel'
import { Sidebar } from './components/Sidebar'
import { StatCard } from './components/StatCard'
import { StepDetailsModal } from './components/StepDetailsModal'
import {
  assistantMessages,
  confidenceData,
  kpis,
  latencyData,
  pipelineSteps,
  recentQueries,
  retrievalEvidence,
  type PipelineStep,
} from './data/mockData'

type CompareApiResponse = {
  outputs?: {
    rag_answer?: string
    non_rag_answer?: string
    ml_prediction?: {
      predicted_priority_label?: string
      predicted_priority?: string
      probabilities?: Record<string, number>
    }
    llm_zero_shot_prediction?: {
      prediction?: {
        priority?: string
        confidence?: number
        rationale?: string
        next_action?: string
      }
    }
    retrieved?: Array<{
      rank?: number
      similarity_score?: number
      question_tweet_id?: string | number
      document_text?: string
      query_text?: string
    }>
  }
  metrics?: {
    latency_ms?: { total?: number }
  }
}

function App() {
  const [selectedStep, setSelectedStep] = useState<PipelineStep | null>(null)
  const [developerMode, setDeveloperMode] = useState(false)
  const [ticketInput, setTicketInput] = useState('urgent payment failed and internet down')
  const [isLoading, setIsLoading] = useState(false)
  const [chatMessages, setChatMessages] = useState(assistantMessages)
  const [chunks, setChunks] = useState(retrievalEvidence)
  const [statItems, setStatItems] = useState(kpis)

  const runInference = async () => {
    if (!ticketInput.trim()) {
      return
    }
    setIsLoading(true)
    setChatMessages((prev) => [...prev, { role: 'user', text: ticketInput }])
    try {
      const res = await fetch('http://127.0.0.1:8000/comparison/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticket: ticketInput,
          retrieval_backend: 'chroma',
          k: 5,
          allow_llm_fallback: true,
          no_log: false,
          log_top_k: 2,
          log_max_text_chars: 120,
          hash_ticket_in_log: true,
        }),
      })
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`)
      }
      const data: CompareApiResponse = await res.json()
      const ragText = data.outputs?.rag_answer ?? 'No RAG answer returned.'
      const nonRagText = data.outputs?.non_rag_answer ?? 'No non-RAG answer returned.'
      const mlPriority = data.outputs?.ml_prediction?.predicted_priority ?? 'N/A'
      const llmPriority = data.outputs?.llm_zero_shot_prediction?.prediction?.priority ?? 'N/A'
      const conf = data.outputs?.llm_zero_shot_prediction?.prediction?.confidence ?? 0
      const totalLatency = data.metrics?.latency_ms?.total ?? 0

      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `RAG: ${ragText}` },
        { role: 'assistant', text: `Non-RAG: ${nonRagText}` },
      ])

      const retrieved = (data.outputs?.retrieved ?? []).slice(0, 3).map((r, idx) => ({
        id: `live_${idx}`,
        source: String(r.question_tweet_id ?? `ticket_${idx + 1}`),
        score: Number(r.similarity_score ?? 0),
        snippet: r.document_text || r.query_text || 'No snippet available',
      }))
      if (retrieved.length > 0) {
        setChunks(retrieved)
      }

      setStatItems([
        { title: 'Predicted Priority', value: String(mlPriority), delta: `LLM: ${llmPriority}`, trend: 'live' },
        { title: 'Model Confidence', value: Number(conf).toFixed(2), delta: 'Zero-shot confidence', trend: 'live' },
        { title: 'Retrieved Chunks', value: String(data.outputs?.retrieved?.length ?? 0), delta: 'Top-k evidence', trend: 'k=5' },
        { title: 'End-to-End Latency', value: `${(Number(totalLatency) / 1000).toFixed(2)}s`, delta: 'API response', trend: 'live' },
      ])
    } catch (error) {
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Request failed. Make sure backend is running on :8000. (${String(error)})` },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-6">
        <div className="mx-auto flex max-w-[1480px] flex-col gap-5">
          <Header
            developerMode={developerMode}
            onToggleDeveloperMode={() => setDeveloperMode((prev) => !prev)}
          />
          <InputPanel
            value={ticketInput}
            isLoading={isLoading}
            onChange={setTicketInput}
            onRun={runInference}
            onUseSample={() => setTicketInput('urgent payment failed and internet down')}
          />
          <PipelineFlow steps={pipelineSteps} onSelectStep={setSelectedStep} />
          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {statItems.map((kpi) => (
              <StatCard key={kpi.title} {...kpi} />
            ))}
          </section>
          <InsightsCharts latencyData={latencyData} confidenceData={confidenceData} />
          <section className="grid gap-4 xl:grid-cols-[1.25fr_1fr]">
            <RetrievalPanel chunks={chunks} />
            <ChatAssistant messages={chatMessages} isLoading={isLoading} />
          </section>
          <RecentQueriesTable rows={recentQueries} />
          <DeveloperModePanel enabled={developerMode} />
        </div>
      </main>
      <StepDetailsModal step={selectedStep} onClose={() => setSelectedStep(null)} />
    </div>
  )
}

export default App
