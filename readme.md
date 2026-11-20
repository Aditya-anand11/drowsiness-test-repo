# <PROJECT_NAME>

## Overview

<PROJECT_NAME> is a production-oriented full-stack TypeScript application. It exposes a versioned HTTP API, serves a React/Next.js web client, stores relational data in PostgreSQL through Prisma, uses Redis for caching and background jobs, and ships with Docker-based local and production workflows.

This README is the canonical architecture and implementation guide for the repository. It documents the high-level architecture, core technology choices, request lifecycle, module boundaries, implementation details, and the responsibility of each canonical source file. Replace angle-bracket placeholders with project-specific values.

The design follows a layered architecture: transport -> validation -> controller -> service -> repository -> persistence. Dependencies point inward. Controllers do not contain business logic. Services do not contain HTTP concerns. Repositories do not contain domain rules.

## Architecture at a glance

```text
Browser / mobile client
        |
        v
CDN + reverse proxy (TLS, compression, static assets)
        |
        +--> Next.js web client (server components + client components)
        |          |
        |          v
        |     API client (typed fetch, auth cookies, error decoding)
        |
        +--> Express API (/api/v1)
                   |
                   +--> request id -> security -> auth -> rate limit -> validation
                   +--> controller -> service -> repository
                   +--> PostgreSQL through Prisma
                   +--> Redis for cache, sessions, and queues
                   +--> worker process for asynchronous jobs
                   +--> object storage for uploads
```

The API is stateless. Sessions or tokens are carried by HTTP-only cookies where possible. Long-running work is delegated to workers. PostgreSQL is the source of truth. Redis is never the source of truth; it is an optimization and coordination layer.

## Core technology

| Area | Choice | Why it is used |
| --- | --- | --- |
| Runtime | Node.js 20 LTS | Stable runtime with strong TypeScript support |
| Language | TypeScript strict mode | Compile-time safety across API, shared schemas, and UI |
| API framework | Express | Small, explicit middleware pipeline and broad ecosystem |
| ORM | Prisma | Typed database access, migrations, and relation handling |
| Database | PostgreSQL | Transactions, constraints, indexes, and relational integrity |
| Cache and queues | Redis + BullMQ | Caching, rate-limit state, sessions, and background jobs |
| Frontend | Next.js App Router + React | Server rendering, routing, and client interactivity |
| Styling | Tailwind CSS | Consistent design tokens and utility-first styling |
| Validation | Zod | Runtime validation with inferred TypeScript types |
| Auth | HTTP-only cookies with short-lived sessions | Reduced token exposure in browser JavaScript |
| Logging | pino | Structured JSON logs with low overhead |
| Testing | Vitest, Supertest, Playwright | Unit, integration, and end-to-end coverage |
| Packaging | Docker and Docker Compose | Reproducible local and production environments |
| CI/CD | GitHub Actions | Linting, type checks, tests, builds, and image publishing |

## Request lifecycle

1. The client sends an HTTPS request to the reverse proxy or CDN.
2. Static assets are served from the edge. API and page requests are forwarded to the application.
3. The API assigns a request id and attaches it to the logger context.
4. Security middleware sets headers, applies CORS rules, and rejects disallowed methods or origins.
5. Auth middleware resolves the session or token and attaches the authenticated principal when present.
6. Rate-limit middleware checks Redis-backed counters for sensitive routes.
7. Validation middleware parses path, query, and body input with Zod schemas.
8. The controller maps validated input to a service call and formats the HTTP response.
9. The service applies domain rules, authorization checks, transactions, and idempotency.
10. The repository performs typed persistence operations through Prisma.
11. PostgreSQL returns data inside a transaction when required.
12. The service may enqueue a background job and return a fast response.
13. The response envelope returns data or a typed error. Errors include the request id for tracing.
14. The worker consumes queued jobs and performs retries with backoff and dead-letter handling.

## Repository layout

```text
.
├── apps
│   ├── api
│   │   ├── src
│   │   │   ├── app.ts
│   │   │   ├── server.ts
│   │   │   ├── config
│   │   │   ├── controllers
│   │   │   ├── db
│   │   │   ├── jobs
│   │   │   ├── middleware
│   │   │   ├── repositories
│   │   │   ├── routes
│   │   │   ├── schemas
│   │   │   ├── services
│   │   │   ├── types
│   │   │   └── utils
│   │   └── prisma
│   └── web
│       ├── app
│       ├── components
│       ├── hooks
│       ├── lib
│       ├── store
│       └── styles
├── packages
│   └── shared
├── tests
│   ├── unit
│   ├── integration
│   └── e2e
├── scripts
├── docs
├── Dockerfile
├── docker-compose.yml
├── package.json
├── tsconfig.json
└── .env.example
```

## Code and module reference

### Root configuration and tooling

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `package.json` | Dependency graph, workspace scripts, and package metadata | Keep scripts small and composable: dev, build, test, lint, typecheck, db:migrate, db:seed. Pin major versions. |
| `tsconfig.json` | TypeScript compiler configuration | Enable strict, noUncheckedIndexedAccess, exactOptionalPropertyTypes, and path aliases where useful. |
| `eslint.config.js` | Static analysis rules | Enforce import order, no-floating-promises, consistent type imports, and framework-specific rules. |
| `vitest.config.ts` | Unit and integration test configuration | Use separate projects or environment settings for node and jsdom tests. |
| `playwright.config.ts` | Browser end-to-end test configuration | Start the app against a disposable database and deterministic seed data. |
| `.env.example` | Documented environment contract | Never contains secrets. Lists every required variable with safe example values. |
| `Dockerfile` | Production image build | Multi-stage build: install dependencies, build, prune dev dependencies, run as a non-root user. |
| `docker-compose.yml` | Local dependency orchestration | Starts PostgreSQL, Redis, the API, the web app, and optionally a worker. Uses health checks and named volumes. |

### API bootstrap

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/server.ts` | Process entry point | Loads configuration, creates the HTTP server, registers graceful shutdown, and starts listening only after the database and Redis are reachable. |
| `apps/api/src/app.ts` | Express application factory | Registers middleware, routes, error handling, and health checks. Exported separately from the server so tests can run without opening a port. |
| `apps/api/src/config/env.ts` | Environment parsing | Validates environment variables with Zod at startup. Fails fast with a readable error when configuration is missing or malformed. |
| `apps/api/src/config/logger.ts` | Structured logging | Creates a pino logger with redaction for authorization headers, cookies, passwords, and tokens. Adds request id child loggers. |
| `apps/api/src/config/constants.ts` | Shared constants | Centralizes pagination defaults, cookie names, cache TTLs, and error codes. Avoids magic strings across modules. |

### Database and persistence

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/db/client.ts` | Prisma client singleton | Reuses a single client in development to survive hot reload. Exposes a small wrapper for transactions and health checks. |
| `apps/api/prisma/schema.prisma` | Data model and relation definitions | Defines models, enums, indexes, unique constraints, and relation behavior. Uses explicit mapping names for join tables. |
| `apps/api/prisma/migrations/*` | Versioned schema changes | Generated by Prisma and reviewed before deployment. Never edited after being applied to shared environments. |
| `apps/api/prisma/seed.ts` | Local and test seed data | Idempotent. Creates deterministic users, roles, and sample resources for development and end-to-end tests. |
| `apps/api/src/db/health.ts` | Database readiness check | Runs a lightweight query and reports latency. Used by health endpoints and startup checks. |

### Middleware

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/middleware/requestId.ts` | Request correlation | Reads an incoming request id or generates one, attaches it to the request, response header, and logger. |
| `apps/api/src/middleware/security.ts` | HTTP hardening | Applies helmet headers, CORS allowlists, body size limits, and content-type checks. |
| `apps/api/src/middleware/auth.ts` | Authentication | Resolves the session cookie or bearer token, verifies signature and expiry, and attaches the principal. Supports optional auth for public routes. |
| `apps/api/src/middleware/authorize.ts` | Authorization | Checks roles, permissions, or resource ownership after authentication. Throws typed forbidden errors. |
| `apps/api/src/middleware/validate.ts` | Input validation | Parses body, query, and params with Zod schemas. Replaces request input with parsed data and rejects unknown fields where appropriate. |
| `apps/api/src/middleware/rateLimit.ts` | Abuse protection | Uses Redis-backed counters for login, registration, password reset, and write-heavy endpoints. Returns retry-after headers. |
| `apps/api/src/middleware/errorHandler.ts` | Central error mapping | Converts AppError instances and unknown failures into a stable error envelope. Logs server errors with the request id. Does not leak stack traces in production. |
| `apps/api/src/middleware/notFound.ts` | Unknown route handling | Returns a typed not-found error instead of an HTML default. |

### Routing and controllers

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/routes/index.ts` | Route composition | Mounts versioned routers under `/api/v1`, health routes, and documentation routes. |
| `apps/api/src/routes/<resource>.routes.ts` | Resource route definitions | Declares method, path, middleware chain, and controller handler. Contains no business logic. |
| `apps/api/src/controllers/<resource>.controller.ts` | HTTP translation layer | Reads validated input, calls the service, maps results to status codes and response envelopes. Keeps handlers thin and testable. |
| `apps/api/src/controllers/health.controller.ts` | Health endpoints | Reports liveness and readiness. Readiness checks database and Redis dependencies. |

### Services and repositories

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/services/<resource>.service.ts` | Domain logic | Applies business rules, authorization checks, transactions, and idempotency. Throws typed AppError instances. Does not import Express types. |
| `apps/api/src/services/auth.service.ts` | Authentication flows | Handles login, logout, session creation, password hashing, token rotation, and password reset. Uses constant-time comparison where relevant. |
| `apps/api/src/repositories/<resource>.repository.ts` | Persistence boundary | Wraps Prisma queries for one aggregate. Accepts and returns domain-friendly types. Includes pagination, filtering, and soft-delete behavior. |
| `apps/api/src/repositories/user.repository.ts` | User persistence | Finds users by email or id, creates users, updates password hashes, and loads roles or permissions. |
| `apps/api/src/repositories/base.repository.ts` | Shared persistence helpers | Provides common pagination, count, and transaction helpers without hiding Prisma semantics. |

### Schemas and shared types

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/schemas/<resource>.schema.ts` | Request validation schemas | Defines create, update, list, and id schemas with Zod. Exports inferred input types. |
| `apps/api/src/schemas/common.schema.ts` | Reusable validation primitives | Pagination, ids, emails, dates, and sort parameters. |
| `apps/api/src/types/<domain>.ts` | Domain types | Internal service and repository types that are not HTTP-specific. |
| `packages/shared/src/*` | Cross-app contracts | Shared Zod schemas, API response types, and constants used by both API and web client. Keep this package framework-agnostic. |

### Utilities

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/utils/errors.ts` | Error taxonomy | Defines AppError, validation errors, unauthorized, forbidden, not found, conflict, and internal errors. Includes stable error codes. |
| `apps/api/src/utils/asyncHandler.ts` | Async route wrapper | Forwards rejected promises to the Express error handler. |
| `apps/api/src/utils/pagination.ts` | Pagination helpers | Normalizes page and limit, builds metadata, and enforces maximum page size. |
| `apps/api/src/utils/crypto.ts` | Hashing and token helpers | Password hashing, random token generation, and safe comparison helpers. |
| `apps/api/src/utils/cache.ts` | Cache abstraction | Typed get, set, delete, and remember helpers with prefixes and TTLs. |
| `apps/api/src/utils/idempotency.ts` | Idempotency support | Stores request keys for write endpoints and returns the original result for duplicates. |

### Jobs and events

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/api/src/jobs/queue.ts` | Queue definitions | Creates BullMQ queues and shared connection options. Exports typed job names and payload schemas. |
| `apps/api/src/jobs/workers/*.ts` | Background workers | Consume jobs, validate payloads, perform retry-safe work, and record failures. Workers are idempotent. |
| `apps/api/src/jobs/scheduler.ts` | Recurring jobs | Registers cron-like jobs for cleanup, reconciliation, and notifications. |
| `apps/api/src/events/*.ts` | Domain events | Decouples side effects such as email, audit logging, and cache invalidation from the request path. |

### Web client

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `apps/web/app/layout.tsx` | Root layout | Loads fonts, global styles, providers, and shared navigation. Keeps server-rendered structure stable. |
| `apps/web/app/page.tsx` | Landing page | Server component that fetches public data and renders the initial view. |
| `apps/web/app/(auth)/*` | Auth pages | Login, registration, and password reset flows. Uses shared Zod schemas and server actions or API calls. |
| `apps/web/app/(dashboard)/*` | Authenticated application area | Uses route groups, layout-level auth checks, and server-side data loading. |
| `apps/web/components/*` | Reusable UI | Presentational and interactive components. Server components by default; client components only when state or browser APIs are required. |
| `apps/web/lib/api.ts` | API client | Centralizes base URL, credentials, headers, response parsing, and error decoding. Returns typed results. |
| `apps/web/lib/auth.ts` | Client auth helpers | Reads session state, handles redirects, and exposes login and logout actions. |
| `apps/web/hooks/*` | Reusable hooks | Data fetching, form state, debouncing, and UI behavior. No business logic that belongs in the API. |
| `apps/web/store/*` | Client state | Holds only UI and session state. Server data is not duplicated into a global store. |
| `apps/web/styles/globals.css` | Design tokens and global CSS | Tailwind layers, CSS variables, focus styles, and reduced-motion support. |

### Tests, scripts, and docs

| File | Responsibility | Implementation notes |
| --- | --- | --- |
| `tests/unit/*` | Unit tests | Test pure functions, schemas, services with mocked repositories, and utilities. |
| `tests/integration/*` | Integration tests | Exercise HTTP routes against a disposable PostgreSQL database. Use Supertest and real middleware. |
| `tests/e2e/*` | End-to-end tests | Playwright tests for critical user journeys against seeded data. |
| `tests/fixtures/*` | Test data | Factories and deterministic fixtures. Avoid shared mutable state between tests. |
| `scripts/*` | Operational scripts | Database reset, seed, migration checks, backup, and smoke-test helpers. |
| `docs/*` | Supporting documentation | API examples, ADRs, runbooks, and incident procedures. |

## Data model

Core entities:

- User: identity, email, password hash, status, timestamps, and optional profile fields.
- Session: server-side session id, user id, expiry, user agent, and revocation state.
- Role and Permission: authorization grants attached to users or groups.
- Resource: the main domain aggregate. Replace with project-specific entities.
- AuditLog: append-only record of sensitive actions with actor, action, target, and request id.
- JobRecord: optional durable record of queued work for observability and replay.

Rules:

- Every table has a primary key, createdAt, and updatedAt.
- Soft deletes use deletedAt only where recovery or audit requirements exist.
- Unique constraints are enforced in the database, not only in application code.
- Foreign keys use restrictive or cascading behavior deliberately.
- Indexes cover foreign keys, common filters, and sort columns.
- Transactions wrap multi-step writes in the service layer.

## API design

Canonical endpoints:

- `GET /api/v1/health/live` returns process liveness.
- `GET /api/v1/health/ready` checks database and Redis readiness.
- `POST /api/v1/auth/login` authenticates and sets an HTTP-only session cookie.
- `POST /api/v1/auth/logout` revokes the current session.
- `GET /api/v1/auth/me` returns the current principal.
- `GET /api/v1/<resource>` lists resources with pagination, filters, and sorting.
- `POST /api/v1/<resource>` creates a resource.
- `GET /api/v1/<resource>/:id` returns one resource.
- `PATCH /api/v1/<resource>/:id` updates a resource.
- `DELETE /api/v1/<resource>/:id` soft-deletes or hard-deletes according to policy.

Response conventions:

- Success responses use a stable envelope containing data and optional metadata.
- List responses include items, page, limit, total, and hasNextPage.
- Error responses include code, message, requestId, and optional field-level details.
- Validation errors return 422 or 400 consistently.
- Auth errors return 401. Authorization errors return 403. Missing resources return 404. Conflicts return 409.

## Implementation details

### Configuration and bootstrapping

Configuration is parsed once at startup. The application refuses to start when required variables are missing or invalid. Secrets are never logged. The server starts only after dependency checks pass. Graceful shutdown stops accepting connections, drains the HTTP server, closes workers and queues, and disconnects Prisma and Redis.

### API layer

Controllers are intentionally thin. They translate HTTP input into service calls and service results into responses. Validation happens before controllers. Error handling happens after controllers. This keeps each layer replaceable and testable.

### Persistence

Repositories own Prisma access for one aggregate. Services own transactions and domain rules. Queries use select projections to avoid over-fetching. Pagination uses stable sort keys. N+1 queries are avoided with explicit includes or batched loaders. Migrations are reviewed and applied in CI and deployment pipelines before the new application version receives traffic.

### Authentication and authorization

Authentication resolves a session cookie or token, verifies expiry and revocation, and attaches a principal. Authorization checks roles, permissions, or ownership after authentication. Cookies are HTTP-only, secure in production, same-site where appropriate, and scoped to the API path. Sessions are rotated after privilege changes. Password reset tokens are single-use and time-bound.

### Validation and errors

Zod schemas are the single source of truth for request shape and inferred types. Unknown fields are rejected on write endpoints. Errors are represented as typed AppError instances with stable codes. The central error handler maps known errors to HTTP status codes and unknown errors to 500 without leaking internals. Every error log includes the request id.

### Background jobs

Writes that trigger slow side effects enqueue jobs instead of blocking the response. Job payloads are validated against schemas. Workers are idempotent and safe to retry. Failures use exponential backoff and eventually reach a dead-letter queue. Job records include request id, actor id, and attempt count for tracing.

### Frontend

Server components fetch initial data and render SEO-friendly markup. Client components handle local interactivity, optimistic updates, and browser APIs. The API client centralizes credentials and error decoding. Shared Zod schemas keep client and server validation aligned. Forms show field-level errors and disable submission while pending. Accessibility, keyboard navigation, and reduced-motion support are required for interactive components.

### Security

- Apply security headers and a strict content security policy where feasible.
- Enforce CORS allowlists instead of wildcard origins in production.
- Rate-limit authentication, registration, password reset, and write endpoints.
- Validate and normalize all input at the boundary.
- Use parameterized queries through Prisma. Never concatenate SQL.
- Store only hashed passwords with a modern memory-hard algorithm.
- Redact secrets, cookies, and authorization headers from logs.
- Use least-privilege database and object-storage credentials.
- Keep dependencies patched and run automated vulnerability scans in CI.

### Observability

- Structured JSON logs with request id, user id when present, route, status, and latency.
- Health endpoints for liveness and readiness.
- Metrics for request rate, error rate, latency, queue depth, job failures, and database pool usage.
- Traces for inbound requests and outbound dependencies where supported.
- Audit logs for authentication, authorization changes, and destructive operations.

### Performance

- Cache read-heavy, low-churn data in Redis with explicit TTLs and invalidation.
- Use database indexes for common filters and sort orders.
- Avoid over-fetching with Prisma select projections.
- Stream or paginate large collections.
- Move slow side effects to workers.
- Use connection pooling and sensible pool limits.
- Measure before optimizing. Keep performance changes behind benchmarks.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `NODE_ENV` | Runtime mode: development, test, or production |
| `PORT` | API port |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `AUTH_SECRET` | Secret used for signing or encrypting session material |
| `SESSION_COOKIE_NAME` | Name of the session cookie |
| `CORS_ORIGINS` | Comma-separated allowlist of trusted origins |
| `API_BASE_URL` | Public API base URL used by the web client |
| `WEB_BASE_URL` | Public web base URL used for redirects and email links |
| `LOG_LEVEL` | Logging verbosity |
| `RATE_LIMIT_WINDOW_MS` | Rate-limit window size |
| `RATE_LIMIT_MAX` | Maximum requests per window for protected routes |
| `OBJECT_STORAGE_*` | Object-storage endpoint, bucket, and credentials |

## Local development

```bash
cp .env.example .env
docker compose up -d postgres redis
npm install
npm run db:migrate
npm run db:seed
npm run dev
```

Useful commands:

```bash
npm run lint
npm run typecheck
npm run test
npm run test:integration
npm run test:e2e
npm run build
```

## Testing strategy

- Unit tests cover pure functions, schemas, utilities, and services with mocked repositories.
- Integration tests cover HTTP routes, middleware, validation, authentication, error mapping, and real database behavior.
- End-to-end tests cover critical user journeys in a browser against seeded data.
- Tests must be deterministic and isolated. Each integration test run uses a disposable database or transaction rollback.
- New behavior requires a failing test first when practical. Bug fixes include a regression test.

## Deployment

1. CI runs lint, typecheck, unit tests, integration tests, and build.
2. Database migrations are applied before the new application version receives traffic.
3. The API and worker images are built from the same source and deployed with health checks.
4. The web client is built and deployed to the hosting platform or served by the Node runtime.
5. Rolling deployment drains old instances through graceful shutdown.
6. Smoke tests hit health endpoints and one read-only API route.
7. Rollback uses the previous image tag and a compatible migration plan.

## Extension guide

To add a new resource:

1. Add or update Prisma models and create a migration.
2. Add Zod schemas for create, update, list, and id inputs.
3. Add a repository for persistence.
4. Add a service for domain rules and transactions.
5. Add a controller for HTTP translation.
6. Add routes and register them under `/api/v1`.
7. Add integration tests for success, validation, auth, and error cases.
8. Add UI routes and components if the resource is user-facing.
9. Add metrics, audit events, and background jobs for side effects.
10. Update this README and any API documentation.

## Glossary

- Aggregate: a consistency boundary around one domain object and its related data.
- Controller: HTTP adapter that translates requests and responses.
- Repository: persistence adapter for one aggregate.
- Service: domain layer that enforces rules and coordinates transactions.
- Idempotency: the property that repeating the same request produces the same result.
- Readiness: whether the application can serve traffic, including dependencies.
- Liveness: whether the process is running.
- Dead-letter queue: storage for jobs that exhausted retries.
