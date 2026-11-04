# Security

Report vulnerabilities privately to walid.sadfi@evolurise.com. You will get an acknowledgement within 72 hours.

Notes for operators:

- Set `RAG_API_TOKEN`; without it the API is open to anyone who can reach it.
- Uploaded files are parsed with `pypdf`, `python-docx` and `lxml`; only expose `/ingest` to trusted users.
- The container runs as a non-root user with a read-only filesystem.
