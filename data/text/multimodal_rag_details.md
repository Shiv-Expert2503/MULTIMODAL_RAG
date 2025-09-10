# 📄 Portfolio Agent – RAG Documentation  

---

## 🏷️ Metadata (embedding anchors + keywords)  
* **Project Name:** Multimodal AI Portfolio Agent
* **Version:** 1.0 (Stateful RAG with Semantic Routing)
* **Primary Goal:** To act as an interactive, conversational AI expert on Shivansh's personal projects and career portfolio.
* **Core Technologies:** Python, FastAPI, `asyncio`, LangChain, Google Gemini, ChromaDB, React.
* **Key Features:** State-aware RAG, multi-layer semantic routing, JSON tree generation, conversational memory, output sanitation.
* **Author:** Shivansh Singh


Hidden metadata (for embeddings only):  
- “Portfolio chatbot”  
- “Shivansh’s AI agent introduction”  
- “Show projects to recruiter”  
- “Tech stack explanation agent”  
- “Self-intro LLM”  

---

## 🔑 Atomic Chunks (retrieval-friendly units)  

### **Overview of the Portfolio Agent**
- Portfolio agent built by **Shivansh Singh**.  
- Core function: **Answer recruiter and HR questions** about projects, skills, GPA, and portfolio.  
- Built using **Retrieval-Augmented Generation (RAG)**.  
- Uses **semantic routing** to detect intent (general vs project-specific).  
- Asynchronous architecture → **FastAPI + async worker queue**.  

---

### **Semantic Router**
- Custom router combines **embedding similarity + keyword fallback**.  
- Accepts queries via **gap-based acceptance rule** (avoids rigid thresholds).  
- **Pronoun resolution**: “his project” → Shivansh’s last-mentioned project.  
- Returns top-3 candidates for debugging.  

---

### **Conversation Memory**
- Memory window size: **6 exchanges**.  
- Stores **chat history for context + pronoun coreference**.  
- Each topic has **separate deque (maxlen=6)**.  

---

### **Ambiguity Detection**
- Ambiguity rules:  
  - If query mentions “projects” but not specific → ambiguous.  
  - If general vs specific scores are close → ambiguous.  
  - If last assistant msg was project-specific → confirm before switching.  

---

### **General Query Anchor**
- Special cluster: `GENERAL_TOPICS`.  
- Includes: “general_query”, “all projects”, “portfolio overview”, “about Shivansh”, “list of projects”, “GPA”.  
- Ensures recruiter queries always route to **structured overview tree**.  

---

### **Prompt Engineering Rules**
- Four formatting modes:  
  1. **Overview** → JSON tree: intro + bullets + optional 1 image + follow-up.  
  2. **Detail** → Pipeline steps, tech stack, results (bullets / tables).  
  3. **Resources Only** → Markdown links only.  
  4. **Image Only** → Markdown image tags only.  
- Guardrails:  
  - No hallucinated links.  
  - No leaking internal dataset.  
  - Consistent recruiter-facing tone.  

---

### **Image Handling**
- Function `extract_image_paths()` finds image references in text.  
- `enrich_text_with_markdown_images()` embeds them as Markdown.  
- Explicit captions for recruiter clarity.  

---

### **Async Infrastructure**
- FastAPI backend.  
- Requests enqueued → worker processes in background.  
- Ensures **scalability & responsiveness**.  

---

### **Caching Layer**
- Embeddings cached with **pickle**.  
- Reloads faster, avoids duplicate API calls.  

---

### **Error Handling & Safety**
- Custom `CustomException` class.  
- If JSON parsing fails, retries generation.  
- Always returns safe defaults.  

---

## 🖼️ Explicit Image Captions  

- **Diagram: Architecture Overview**  
  _Shows semantic router, memory module, async queue, and RAG retriever flow._  

- **Screenshot: Recruiter Query Example**  
  _Example query: “Tell me about Shivansh’s projects” → Agent responds with structured JSON tree._  

- **Screenshot: Ambiguity Detection Flow**  
  _User asks “Tell me about his work” → Agent confirms reference before answering._  

---

## ❓ FAQ-style Redundancy (retrieval anchors)  

**Q: What is this project?**  
A: An AI portfolio agent built by Shivansh Singh to answer recruiter and HR questions about his portfolio using RAG.  

**Q: How does it handle vague questions like “his project”?**  
A: Uses conversation memory + pronoun rewriting to map “his” → Shivansh’s projects.  

**Q: What happens if I ask “projects” without details?**  
A: Ambiguity detector triggers → responds with structured overview of all projects.  

**Q: Can it show images of projects?**  
A: Yes, images are auto-embedded from documentation into Markdown with captions.  

**Q: What tech stack powers the agent?**  
A: Python, FastAPI, AsyncIO, semantic embeddings, custom router, RAG retrieval, and pickle-based caching.  

**Q: Who is the intended audience?**  
A: Recruiters, HRs, and interviewers looking at Shivansh’s portfolio.  

---

## 🔗 Synonyms & Cross-links  

- “Show projects” → “list projects”, “all projects”, “portfolio overview”.  
- “GPA” → “grades”, “marks”, “academic score”.  
- “Tell me about Shivansh” → “about the candidate”, “who is Shivansh”, “profile summary”.  
- “Explain a project” → “project details”, “pipeline explanation”, “technical deep dive”.  

Cross-links:  
- From **General Query** → Overview tree → **Detail Mode** for each project.  
- From **FAQ answers** → Linked back to semantic anchors (“general_query”, “project detail”).  

---

## 📊 Structured Recap  

### **Technical Features**
| Module              | Function                                                                 |
|---------------------|--------------------------------------------------------------------------|
| Semantic Router     | Embedding + keyword + gap acceptance                                     |
| Conversation Memory | Stores last 6 exchanges, pronoun resolution                              |
| Ambiguity Detection | Identifies vague queries, confirms before answering                      |
| General Topics      | Anchors recruiter queries → overview mode                                |
| Prompt Rules        | Enforces recruiter-style responses (overview, detail, resources, image)  |
| Image Handling      | Auto-embeds Markdown images with captions                                |
| Async Queue         | FastAPI worker for scalable async requests                               |
| Cache Layer         | Pickle-based embedding cache                                             |

---

### **Recruiter Value Recap**
- Clear, structured answers → avoids fluff.  
- Handles **both broad and detailed** queries.  
- Prevents hallucinations → recruiter trust.  
- Visual + textual → images enrich explanation.  
- Async infra → deployable at scale.  
