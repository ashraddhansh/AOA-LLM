# Chroma embedding migrations

Chroma collections have a fixed embedding dimension. The previous Google
embedding setup created `aoa_documents` with 3072-dimensional vectors. The
current Groq-compatible setup uses the local LlamaIndex hash embedder with
256-dimensional vectors, so it stores new documents in:

```text
aoa_documents_local_v2
```

The collection can be changed with `CHROMA_COLLECTION_NAME`. Do not point the
current 256-dimensional embedder at the old `aoa_documents` collection.
When changing embedding models, use a new collection name or recreate the
database after preserving any data that must be migrated.
