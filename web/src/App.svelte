<script>
  // The shell: a token, five views, and a live event strip.
  //
  // Deliberately plain. Every number on these pages is the service's, every
  // permission decision is the service's, and the UI's whole job is to show the
  // trade-offs the system is making without flattening them into a score.
  import Served from './views/Served.svelte'
  import Swarms from './views/Swarms.svelte'
  import Promotions from './views/Promotions.svelte'
  import Allocation from './views/Allocation.svelte'
  import Capacity from './views/Capacity.svelte'
  import Operations from './views/Operations.svelte'
  import { health, setToken, token, subscribe } from './lib/api.js'

  const views = [
    ['served', 'Served surface', Served],
    ['swarms', 'Swarms', Swarms],
    ['promotions', 'Promotions', Promotions],
    ['allocation', 'Allocation', Allocation],
    ['capacity', 'Capacity', Capacity],
    ['operations', 'Operations', Operations],
  ]

  let current = (location.hash || '#served').slice(1)
  let entered = token()
  let status = null
  let events = []
  let feedError = null

  function go(name) {
    current = name
    location.hash = `#${name}`
  }

  async function connect() {
    setToken(entered)
    try {
      status = await health()
    } catch (error) {
      status = { ok: false, error: error.message }
    }
  }

  connect()

  // The live strip. Events only — no payloads, because a `no_sample` run has none
  // and there is no privileged view that would show one anyway.
  const stop = subscribe(
    (event) => {
      events = [event, ...events].slice(0, 12)
      feedError = null
    },
    (error) => {
      feedError = error.message || 'the event feed dropped; the browser will retry'
    },
  )

  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', stop)
    window.addEventListener('hashchange', () => {
      current = (location.hash || '#served').slice(1)
    })
  }

  $: view = views.find(([name]) => name === current) ?? views[0]
</script>

<header class="top">
  <h1>agent-system</h1>
  <nav>
    {#each views as [name, label]}
      <button class:active={name === current} on:click={() => go(name)}>{label}</button>
    {/each}
  </nav>
  <div class="session">
    <input
      type="password"
      placeholder="token"
      bind:value={entered}
      on:change={connect}
    />
    {#if status}
      {#if status.ok}
        <span class="ok" title={`node ${status.node}`}>
          {status.profile}{status.needs_external_services ? '' : ' · self-contained'}
        </span>
      {:else}
        <span class="error">{status.error ?? 'unhealthy'}</span>
      {/if}
    {/if}
  </div>
</header>

{#if status && status.kill_switches > 0}
  <p class="banner">
    {status.kill_switches} kill switch{status.kill_switches === 1 ? '' : 'es'} engaged.
    Dispatch is stopped in its scope.
  </p>
{/if}

<main>
  <svelte:component this={view[2]} />
</main>

<aside>
  <h3>Recent events</h3>
  {#if feedError}
    <p class="muted">{feedError}</p>
  {/if}
  {#if !events.length}
    <p class="muted">Nothing yet.</p>
  {:else}
    <ul>
      {#each events as event}
        <li>
          <code>{event.kind}</code>
          <span class="muted">{event.shard}#{event.seq}</span>
        </li>
      {/each}
    </ul>
  {/if}
  <p class="muted small">
    Payloads appear only where one was kept and has not expired. A
    <code>no_sample</code> run shows content-free fields only — for every role,
    including admin, because the payload does not exist.
  </p>
</aside>

<style>
  :global(:root) {
    --fg: #15181c;
    --bg: #fbfbfc;
    --muted: #6a7280;
    --line: #dcdfe4;
    --chip: #eceef2;
    --accent: #2b5fd9;
    --warn: #9a6400;
    --bad: #b3261e;
    --ok: #1a7f37;
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :global(:root) {
      --fg: #e6e8eb;
      --bg: #14171a;
      --muted: #9aa3ad;
      --line: #2c3238;
      --chip: #22272c;
      --accent: #7aa2f7;
      --warn: #d9a400;
      --bad: #f06a60;
      --ok: #4cc26a;
    }
  }
  :global(body) {
    margin: 0;
    font: 14px/1.5 system-ui, sans-serif;
    color: var(--fg);
    background: var(--bg);
    display: grid;
    grid-template-columns: 1fr 20rem;
    grid-template-rows: auto 1fr;
    min-height: 100vh;
  }
  :global(table) {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5rem 0 1.5rem;
  }
  :global(th),
  :global(td) {
    border-bottom: 1px solid var(--line);
    padding: 0.35rem 0.5rem;
    text-align: left;
    vertical-align: top;
  }
  :global(th) {
    font-weight: 600;
    font-size: 0.85em;
    color: var(--muted);
    text-transform: lowercase;
  }
  :global(code) {
    font-family: ui-monospace, monospace;
    font-size: 0.92em;
  }
  :global(button) {
    font: inherit;
    padding: 0.2rem 0.6rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--chip);
    color: inherit;
    cursor: pointer;
  }
  :global(button:disabled) {
    opacity: 0.5;
    cursor: default;
  }
  :global(input),
  :global(select) {
    font: inherit;
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--bg);
    color: inherit;
  }
  :global(.muted) {
    color: var(--muted);
  }
  :global(.ok) {
    color: var(--ok);
  }
  :global(.error) {
    color: var(--bad);
  }
  .top {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--line);
  }
  .top h1 {
    font-size: 1rem;
    margin: 0;
  }
  nav {
    display: flex;
    gap: 0.25rem;
  }
  nav button.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .session {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .banner {
    grid-column: 1 / -1;
    margin: 0;
    padding: 0.5rem 1rem;
    background: var(--bad);
    color: white;
  }
  main {
    padding: 1rem;
    min-width: 0;
  }
  aside {
    border-left: 1px solid var(--line);
    padding: 1rem;
  }
  aside ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  aside li {
    padding: 0.15rem 0;
  }
  .small {
    font-size: 0.85em;
  }
</style>
