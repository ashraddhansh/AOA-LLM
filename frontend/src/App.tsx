import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  Check,
  FileText,
  Menu,
  MessageCircle,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";

type Verification = {
  passed: boolean;
  verified: boolean;
  score: number | null;
  explanation: string | null;
};
type Remediation = {
  failure_type?: string | null;
  remediation?: string | null;
  source_label?: string;
  remediation_error?: string | null;
};
type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  verification?: Verification;
  remediation?: Remediation;
  sources?: { text: string; score: number }[];
};
type Chat = {
  id: string;
  title: string;
  updatedAt: string;
  documents: string[];
  messages: Message[];
};

const initialChats: Chat[] = [
  {
    id: "product-notes",
    title: "Product notes",
    updatedAt: "Just now",
    documents: ["product-brief.pdf", "research-notes.docx"],
    messages: [
      {
        id: "welcome",
        role: "assistant",
        content:
          "I’m ready to answer from your documents. Ask a question and I’ll cite only the context I can verify.",
      },
    ],
  },
  {
    id: "policy-review",
    title: "Policy review",
    updatedAt: "Yesterday",
    documents: ["employee-handbook.pdf"],
    messages: [],
  },
];

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "");

type ApiChat = {
  id: string;
  title: string;
  updated_at: string;
  documents: string[];
  message_count: number;
};

const fromApiChat = (chat: ApiChat): Chat => ({
  id: chat.id,
  title: chat.title,
  updatedAt: new Date(chat.updated_at).toLocaleDateString(),
  documents: chat.documents,
  messages: [],
});

function VerificationCard({
  result,
  remediation,
}: {
  result?: Verification;
  remediation?: Remediation;
}) {
  if (!result) {
    return <small className="verified"><Check size={12} /> Verification pending</small>;
  }

  const status = !result.verified
    ? "Unverified"
    : result.passed
      ? "Passed"
      : "Failed";
  const statusClass = !result.verified
    ? "verification-unknown"
    : result.passed
      ? "verification-pass"
      : "verification-fail";
  const score =
    result.score === null ? "—" : `${Math.round(result.score * 100)}%`;

  return (
    <div className={`verification-card ${statusClass}`}>
      <div className="verification-header">
        <span className="verification-title">
          <ShieldCheck size={14} /> Lynx verification
        </span>
        <strong>{status}</strong>
      </div>
      <div className="verification-meta">
        <span>Faithfulness score</span>
        <b>{score}</b>
      </div>
      {result.explanation ? (
        <p className="verification-explanation">{result.explanation}</p>
      ) : null}
      {remediation?.remediation ? (
        <p className="verification-explanation">
          {remediation.remediation === "web_search"
            ? "Answer from the web — not found in your documents."
            : remediation.remediation === "strict_regeneration"
              ? "Answer from your documents — document-grounded."
              : "I could not find reliable information to answer this."}
        </p>
      ) : null}
      {remediation?.remediation_error ? (
        <p className="verification-explanation">
          Remediation detail: {remediation.remediation_error}
        </p>
      ) : null}
    </div>
  );
}

function App() {
  const [chats, setChats] = useState<Chat[]>(initialChats);
  const [activeId, setActiveId] = useState(initialChats[0].id);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [isUploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const activeChat = chats.find((chat) => chat.id === activeId) ?? chats[0];

  useEffect(() => {
    if (!apiBaseUrl) return;
    void fetch(`${apiBaseUrl}/chats`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Unable to load chats");
        return (await response.json()) as ApiChat[];
      })
      .then((remoteChats) => {
        if (!remoteChats.length) return createRemoteChat();
        const nextChats = remoteChats.map(fromApiChat);
        setChats(nextChats);
        setActiveId(nextChats[0].id);
      })
      .catch(() => {
        setError(
          "The FastAPI service is unavailable. Start it on http://localhost:8000 and reload.",
        );
      });
  }, []);

  const createRemoteChat = async () => {
    if (!apiBaseUrl) return;
    const response = await fetch(`${apiBaseUrl}/chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New conversation" }),
    });
    if (!response.ok) throw new Error("Unable to create chat");
    const chat = fromApiChat((await response.json()) as ApiChat);
    setChats([chat]);
    setActiveId(chat.id);
  };

  const visibleChats = useMemo(
    () =>
      chats.filter((chat) =>
        chat.title.toLowerCase().includes(search.toLowerCase()),
      ),
    [chats, search],
  );

  const updateActiveChat = (update: (chat: Chat) => Chat) => {
    setChats((current) =>
      current.map((chat) => (chat.id === activeId ? update(chat) : chat)),
    );
  };

  const createChat = async () => {
    if (apiBaseUrl) {
      try {
        await createRemoteChat();
        setSidebarOpen(false);
        return;
      } catch {
        return;
      }
    }
    const id = `chat-${Date.now()}`;
    setChats((current) => [
      {
        id,
        title: "New conversation",
        updatedAt: "Just now",
        documents: [],
        messages: [],
      },
      ...current,
    ]);
    setActiveId(id);
    setSidebarOpen(false);
  };

  const sendQuestion = async (event: FormEvent) => {
    event.preventDefault();
    const question = query.trim();
    if (!question || !activeChat) return;
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
    };
    setQuery("");
    updateActiveChat((chat) => ({
      ...chat,
      title: chat.messages.length === 0 ? question.slice(0, 34) : chat.title,
      updatedAt: "Just now",
      messages: [...chat.messages, userMessage],
    }));

    if (apiBaseUrl) {
      try {
        const response = await fetch(`${apiBaseUrl}/chats/${activeId}/questions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: question }),
        });
        if (!response.ok) {
          const detail = (await response.json().catch(() => null)) as {
            detail?: string;
          } | null;
          throw new Error(detail?.detail || "Question request failed");
        }
        const data = (await response.json()) as {
          answer: string;
          sources: { text: string; score: number }[];
          verification: Verification;
          failure_type?: string | null;
          remediation?: string | null;
          source_label?: string;
          remediation_error?: string | null;
        };
        updateActiveChat((chat) => ({
          ...chat,
          messages: [
            ...chat.messages,
            {
              id: `assistant-${Date.now()}`,
              role: "assistant",
              content: data.answer,
              verification: data.verification,
              remediation: {
                failure_type: data.failure_type,
                remediation: data.remediation,
                source_label: data.source_label,
                remediation_error: data.remediation_error,
              },
              sources: data.sources,
            },
          ],
        }));
        return;
      } catch (requestError) {
        updateActiveChat((chat) => ({
          ...chat,
          messages: [
            ...chat.messages,
            {
              id: `assistant-${Date.now()}`,
              role: "assistant",
              content: requestError instanceof Error
                ? requestError.message
                : "The RAG service could not answer this question.",
            },
          ],
        }));
        return;
      }
    }

    window.setTimeout(() => {
      updateActiveChat((chat) => ({
        ...chat,
        messages: [
          ...chat.messages,
          {
            id: `assistant-${Date.now()}`,
            role: "assistant",
            content:
              "I don’t know from the uploaded documents yet. Add a relevant source or try a more specific question.",
          },
        ],
      }));
    }, 450);
  };

  const uploadDocuments = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;
    setUploading(true);
    setError("");
    if (apiBaseUrl) {
      const body = new FormData();
      files.forEach((file) => body.append("files", file));
      try {
        const response = await fetch(`${apiBaseUrl}/chats/${activeId}/documents`, {
          method: "POST",
          body,
        });
        if (!response.ok) throw new Error("Upload failed");
      } catch {
        setUploading(false);
        setError(
          "Upload failed. Confirm the FastAPI server is running and that GROQ_API_KEY is configured.",
        );
        return;
      }
    }
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    updateActiveChat((chat) => ({
      ...chat,
      documents: [...chat.documents, ...files.map((file) => file.name)],
    }));
    setUploading(false);
    event.target.value = "";
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${isSidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={19} /></div>
          <span>AOA</span>
          <span className="brand-label">Answer or Abstain</span>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><X size={18} /></button>
        </div>
        <button className="new-chat" onClick={createChat}><Plus size={18} /> New chat</button>
        <label className="search-box"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search chats" /></label>
        <div className="chat-list">
          <p className="eyebrow">Your chats</p>
          {visibleChats.map((chat) => (
            <button className={`chat-item ${chat.id === activeId ? "selected" : ""}`} key={chat.id} onClick={() => { setActiveId(chat.id); setSidebarOpen(false); }}>
              <MessageCircle size={16} />
              <span><strong>{chat.title}</strong><small>{chat.updatedAt}</small></span>
            </button>
          ))}
        </div>
        <div className="sidebar-footer"><div className="status-dot" /> <span>Local workspace</span><span className="demo-pill">{apiBaseUrl ? "API" : "DEMO"}</span></div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Menu size={20} /></button>
          <div><p className="eyebrow">Workspace / {activeChat?.title}</p><h1>{activeChat?.title}</h1></div>
          <div className="topbar-actions"><span className="secure-label"><ShieldCheck size={15} /> Grounded answers</span></div>
        </header>

        <section className="content">
          {error ? <div className="api-error" role="alert">{error}</div> : null}
          <div className="document-strip">
            <div className="strip-heading"><BookOpen size={18} /><div><strong>Source documents</strong><span>{activeChat?.documents.length ? `${activeChat.documents.length} files in this chat` : "Add documents to get started"}</span></div></div>
            <input ref={fileInput} type="file" multiple accept=".pdf,.docx,.txt,.md,.csv" hidden onChange={uploadDocuments} />
            <button className="upload-button" onClick={() => fileInput.current?.click()} disabled={isUploading}><Upload size={16} /> {isUploading ? "Uploading..." : "Upload files"}</button>
          </div>
          {activeChat?.documents.length ? <div className="document-list">{activeChat.documents.map((document) => <span className="document-chip" key={document}><FileText size={14} />{document}<Check size={13} /></span>)}</div> : null}

          <div className="conversation">
            {activeChat?.messages.length === 0 ? <div className="empty-state"><div className="empty-icon"><Sparkles size={25} /></div><h2>Ask your documents anything</h2><p>Upload a source above, then ask a question. AOA retrieves relevant passages and abstains when the answer cannot be verified.</p><div className="suggestions"><button onClick={() => setQuery("Summarize the key points")}>Summarize the key points <ArrowUp size={14} /></button><button onClick={() => setQuery("What are the main risks?")}>What are the main risks? <ArrowUp size={14} /></button></div></div> : activeChat?.messages.map((message) => <article className={`message ${message.role}`} key={message.id}><div className="avatar">{message.role === "assistant" ? <ShieldCheck size={17} /> : "You"}</div><div><span className="message-label">{message.role === "assistant" ? "AOA" : "You"}</span><p>{message.content}</p>{message.role === "assistant" ? <VerificationCard result={message.verification} remediation={message.remediation} /> : null}{message.role === "assistant" && message.sources?.length ? <details className="source-details"><summary>Search/context sources ({message.sources.length})</summary>{message.sources.map((source, index) => <p className="source-item" key={`${source.text}-${index}`}>{source.text}</p>)}</details> : null}</div></article>)}
          </div>

          <form className="composer" onSubmit={sendQuestion}><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendQuestion(event); } }} placeholder="Ask a question about your documents..." rows={1} /><button type="submit" disabled={!query.trim()} aria-label="Send question"><ArrowUp size={19} /></button><span className="composer-hint">Enter to send · Shift + Enter for a new line</span></form>
          <p className="disclaimer">AOA answers only from your uploaded sources. It will say “I don’t know” when the documents do not provide enough evidence.</p>
        </section>
      </main>
    </div>
  );
}

export default App;
