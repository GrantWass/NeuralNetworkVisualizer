# Neural Network Visualizer

Interactive visualizations of neural networks and transformers.

- **`backend/`** — a pure-NumPy neural network implementation (`NeuralNetwork`, `NeuronLayer`, optimizers) exposed through a FastAPI app. Trains on small datasets (XOR, iris, auto_mpg, MNIST) and streams per-layer activations/gradients to the frontend. Deployed to AWS Lambda via Mangum.
- **`neural-network-visual/`** — the Next.js frontend: network training visualizer, attention visualizer, and live GPT-2 inference views.

## Backend development

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

Run tests and lint:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

Run the API locally without any AWS dependencies:

```bash
SESSION_BACKEND=local .venv/bin/uvicorn app:app --reload
```

### Environment variables

| Variable           | Default | Description                                                                                          |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------------- |
| `SESSION_BACKEND`  | `aws`   | `local` stores sessions/leaderboard in-process (dev/tests); `aws` uses DynamoDB + S3                  |
| `ALLOWED_ORIGINS`  | `*`     | Comma-separated CORS origins, e.g. `https://yoursite.com,http://localhost:3000`. Credentials are only enabled for explicit origins |

In AWS mode, sessions are stored as JSON in S3 (`nn-sessions-data`) indexed by TTL'd DynamoDB rows (`nn-sessions`), the leaderboard lives in DynamoDB (`nn-leaderboard`) with conditional writes so concurrent submissions can't clobber each other, and assignments/submissions live in `nn-assignments` / `nn-submissions`.

### Classroom assignments (backend)

Assignments are server-authoritative: scores are computed by evaluating the
student's stored network on the held-out test set, and epoch caps are enforced
against a per-session epoch counter kept in session state.

| Endpoint | Description |
| --- | --- |
| `POST /assignments/create` | Instructor creates `{title, dataset, metric_target, epoch_cap}` → returns join code + secret `instructor_key` (shown once) |
| `GET /assignments/{code}` | Public assignment metadata for students |
| `POST /assignments/{code}/submit` | `{session_id, student_name}` → server-verified score; keeps best score per student atomically |
| `GET /assignments/{code}/submissions?instructor_key=…` | Roster, sorted, requires the instructor key |

Note: adding backend endpoints also requires registering them on the API
Gateway HTTP API (route → Lambda integration) **and** granting the gateway
invoke permission with a source ARN of the form
`arn:aws:execute-api:<region>:<account>:<api-id>/*/*/<path>` — no method
segment.

## Frontend development

```bash
cd neural-network-visual
npm install
npm run dev
```

Lint and build checks:

```bash
npm run lint
npm run build
```

## CI

GitHub Actions runs backend lint + tests and frontend lint + build on every push/PR (see `.github/workflows/ci.yml`).
