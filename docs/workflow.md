# E3-SSE Development Workflow

The project uses the same repository name everywhere:

- Local VS Code workspace: `e3sse`
- GitHub: `github.com/SirHarginger/e3sse`
- Server: `/srv/ben/e3sse`

## Workflow

1. Develop and test locally in VS Code.
2. Commit work to a feature branch.
3. Push to GitHub.
4. Merge approved work into `main`.
5. Pull `main` on the server.
6. Run full-data calculations on the server.
7. Keep raw data, logs, scratch files and generated outputs outside Git.
