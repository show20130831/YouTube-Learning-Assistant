# Labeling guidelines

Labels say which of the user's topics a video **substantively discusses**. They are the reference answer for
topic precision and recall, so they are written from the transcript only, before any model output for the video
exists, and every label carries a short quote from the transcript as evidence.

## General rules

- **Substantive, not a mention.** A topic is labeled when a meaningful part of the video (roughly a few minutes,
  or a central point of a short video) is about it. A passing mention, a sponsor slot or a list of names does not
  count.
- **Transcript only.** Judge from what is said, not from the title, the thumbnail or what you know about the
  channel. The title can point to where to look, but the evidence must come from the transcript.
- **Evidence.** For each labeled topic, quote one or two transcript lines (up to about 200 characters) that show
  the video discusses it. If you cannot find such a line, do not label the topic.
- **No topic is a valid answer.** A video can have no topic from the list.
- **Uncertain cases** go in `notes` with the reason, so the person confirming the labels can decide.
- **LLM vs Machine Learning.** The internals of large language models (attention, KV cache, decoding) are
  labeled LLM only. Machine Learning is added when general methods are taught (training neural networks,
  fine-tuning steps such as validation loss and checkpoints). A small language model built from scratch is
  Machine Learning, not LLM.

## Topics

| Topic | Label when the video discusses... | Not enough on its own |
|---|---|---|
| AI Agent | Systems where a model plans and acts over several steps with tools: building, evaluating, deploying or comparing agents (including coding agents used as agents) | Using a chatbot to answer a question |
| LLM | Large language models themselves: capabilities, comparisons between models, how they work inside (architecture, inference, e.g. KV cache), prompting them, their cost or limits | An LLM only as an invisible engine of a product being demonstrated |
| RAG | Retrieving documents or data and giving them to a model to ground its answer | Plain search, or "the model knows" |
| LangChain | LangChain's framework or its company's products (e.g. LangSmith, Deep Agents) | Naming LangChain in a list of tools |
| LangGraph | LangGraph specifically | LangChain products other than LangGraph |
| AI Automation | Using AI to automate work or business processes end to end (workflows, scheduled or triggered AI jobs, replacing manual steps) | One-off use of an AI tool |
| Data Engineering | Data pipelines, ETL, data infrastructure, storage or processing at scale | Analysing a single dataset |
| Machine Learning | Machine learning methods: training models, neural networks, evaluation of learned models, the statistics behind them when tied to ML | General statistics that is not about learning from data (e.g. multiple-testing corrections) |
| Bioinformatics | Biological or genomic data analysis | Using biology only as an example |

## Process

1. Labels are drafted from the transcript, with evidence, and marked `status: draft`.
2. A person checks each label against its evidence (opening the transcript when the evidence is not convincing)
   and sets `status: confirmed`.
3. Once a dataset version has been used for a run, its labels are frozen. Corrections go into a new dataset
   version.
