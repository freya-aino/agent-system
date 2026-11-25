<script>
  // One place that knows how to be loading, how to have failed, and how to poll.
  //
  // The error is rendered with what the service said about retrying, because
  // "overloaded, try again" and "forbidden" are different situations and a single
  // "something went wrong" would hide which one this is.
  export let load
  export let every = 0

  let state = { status: 'loading' }
  let timer = null

  async function run() {
    try {
      const data = await load()
      state = { status: 'ok', data }
    } catch (error) {
      state = { status: 'error', error }
    }
  }

  $: if (load) {
    run()
    if (timer) clearInterval(timer)
    if (every > 0) timer = setInterval(run, every)
  }

  export function refresh() {
    return run()
  }
</script>

{#if state.status === 'loading'}
  <p class="muted">loading…</p>
{:else if state.status === 'error'}
  <p class="error">
    {state.error.message}
    {#if state.error.status}<span class="muted">({state.error.status})</span>{/if}
    {#if state.error.retryable}
      <button on:click={run}>retry</button>
    {/if}
  </p>
{:else}
  <slot data={state.data} refresh={run} />
{/if}

<style>
  .error {
    color: var(--bad);
  }
  .muted {
    color: var(--muted);
  }
</style>
