import { SendHorizontal } from 'lucide-react'

type Message = {
  role: 'user' | 'assistant'
  text: string
}

type ChatAssistantProps = {
  messages: Message[]
  isLoading?: boolean
}

export function ChatAssistant({ messages, isLoading = false }: ChatAssistantProps) {
  return (
    <section className="glass-card flex h-full flex-col p-5">
      <p className="section-title">AI Assistant</p>
      <div className="mt-4 flex-1 space-y-3 overflow-y-auto">
        {messages.map((msg, idx) => (
          <div
            key={`${msg.role}-${idx}`}
            className={`max-w-[90%] rounded-xl px-3 py-2 text-sm ${
              msg.role === 'assistant'
                ? 'bg-slate-800/80 text-slate-100'
                : 'ml-auto bg-indigo-600/30 text-indigo-100'
            }`}
          >
            {msg.text}
          </div>
        ))}
        {isLoading && (
          <div className="max-w-[90%] rounded-xl bg-slate-800/80 px-3 py-2 text-sm text-slate-100">
            Thinking...
          </div>
        )}
      </div>
      <div className="mt-4 flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-2">
        <input
          type="text"
          placeholder="Ask why this was classified as P1..."
          className="w-full bg-transparent text-sm text-slate-200 outline-none placeholder:text-slate-500"
        />
        <button className="rounded-lg bg-indigo-600 p-2 text-white">
          <SendHorizontal size={14} />
        </button>
      </div>
    </section>
  )
}
