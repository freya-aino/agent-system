# Frontend dependencies

The rule from `docs/stack.md`: npm packages do **not** count against the 30-crate
Rust budget, are kept minimal, and are listed here with a reason. The same
discipline applies — a dependency is justified in one line or it is not added.

| # | Package | Version | Why |
|---|---------|---------|-----|
| 1 | `svelte` | 4.2.19 | The component model. Compiles to plain DOM updates with no runtime framework shipped, which is what keeps the built bundle small enough for S1 to serve from disk without a CDN. |
| 2 | `vite` | 5.4.11 | The build. One config, no plugin stack, and a dev server that proxies to S1 so the UI is developed against the real API rather than a mock. |
| 3 | `@sveltejs/vite-plugin-svelte` | 3.1.2 | Joins the two above. Not optional; not a separate decision. |
| 4 | `svelte-check` | 3.8.6 | Type and template checking in CI. Dev-only, never shipped. |

**Runtime dependencies: none.** Every version is exact rather than a range, for the
same reason the Rust manifests pin: a build that resolves differently on another
machine is a different artifact.

## Deliberately absent

- **No component library.** The views here are tables, a diff and three charts. A
  component library would be larger than the application.
- **No charting library.** The capacity and distribution charts are inline SVG
  (`src/lib/Spark.svelte`, `src/lib/Distribution.svelte`): a few dozen lines each,
  against ~100 kB of dependency.
- **No state-management library.** The API is the state; the UI holds a fetch
  result and a poll interval.
- **No router.** Five views selected by a hash fragment, so a deep link works and
  S1's asset fallback needs no server-side route table.
- **No authentication library.** S1 issues the token; the UI holds it in
  `sessionStorage` and sends it as a bearer. There is nothing to decode client-side,
  because every permission decision is the server's.
