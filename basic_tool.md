## Basic AI Tool Guidance

* **Notebooks** (e.g., Jupyter / Colab)
  * *Use for:* quick experiments, data exploration, and early prototypes.
  * *Don’t use for:* producing finalized products.
  * While you can prototype a chatbot in a notebook (and even spin up a quick Gradio/Streamlit UI), you don’t usually deliver a final product as a notebook in the real world.

* **IDEs** (e.g., VS Code / PyCharm)
  * *Use for:* building final products and real applications (with folders, files, tests, Git, packaging, deployment, UIs, and users). This is the ideal kind of tool in which to build a chatbot or other finalized AI product.
  * *Optional:* Add a coding agent (e.g., Codex) in your IDE; it can read/modify code. IDEs and coding agents are professional-grade helper tools, not a final product themselves. 

* **Commercial apps/websites** (E.g., ChatGPT / Co-pilot / Gemini / NotebookLM)
  * *Use for:* exploring, drafting, brainstorming, coding help.
  * *Don’t use for:* final products. For final products, *do* use an IDE to build your own UI and backend code for calling APIs and other AI tools.
  * The ChatGPT website is its own UI. Don’t try to “wrap” that UI for a product--that's silly.  

* **Coding agents** (e.g., Codex)
  * *Use for:* generating code, fixing bugs, writing tests, etc.
  * *Don’t use for:* hosting or delivering a user-facing final product.
  * These help produce code for your app/product; they are not the app you deliver to end users. Don’t try to wrap coding agent UIs for your products--that's silly.

* **Language model APIs** (e.g., OpenAI API)
  * *Use for:* your app’s backend data processing—chat generation, retrieval, text-to-code, etc. You build your final product using APIs.
  * Language model APIs also mean the difficult computation occurs on **someone else's machine.**
  * Language model APIs are a general low-level tool for interacting with AI models within your own custom-built applications or products.

* **Websites, open-source models + tooling** (e.g., Hugging Face & `transformers` or `langchain` packages) 
  * *Use for:* loading and running open source models; your app’s backend data processing—chat generation, retrieval, text-to-code.
  * Open source models tend to run **on your own machine** (you will need better equipment, more memory, GPUs, etc.)
  * You can build a final product around/with Hugging Face models with or without commerical language model APIs.
  * *Don’t use for:* “wrapping” the Hugging Face website as your own product--that's silly.

* **Agentic tools** (e.g., Zapier, n8n)
  * *Use for:* building automated workflows for your own or small group use.
  * *Don't use for:* building a final product to sell to external users. Like ChatGPT and/or Codex, these are already finalized commerical products with their own UIs.
  * Mostly focused on software engineering automation, but can also perform some office tasks.  

* **Connectors** (e.g., MCP, ChatGPT Apps)
  * Use the model context protocol (MCP) to connect a language model API to various tools (databases, APIs, files, search, internal systems) in a programmatic way; MCP is typically used on the backend of apps or products. 
  * Use specific connecters (e.g., ChatGPT Apps) to let ChatGPT or other commerical apps connect to files, repos, messages, etc.; This enhances ChatGPT's abilities, and is not for building apps or products you surface to others. 

**Usage Rules of Thumb**:
  * *To learn and research for yourself*: Use chatbots (e.g., ChatGPT, NotebookLM). 
  * *To build a product for others to use:* Prototype/test/understand coding logic, API calls, and model usage in notebooks → Build a finalized chatbot or other AI app in an IDE, while letting coding agents help write code and build project infrastructure.
  * *To build an automated workflow for yourself:*
    * *Simple*: Use Zapier, n8n, etc., or ask a chatbot--you may get lucky. 
    * *Advanced*: Use a coding agent to write the software to complete the workflow.
  * Never input sensitive data.
  * Don’t draft, don’t code, prompt!

*Disclosure:* I worked with AI to build these pointers.
