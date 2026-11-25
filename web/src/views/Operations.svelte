<script>
  // Experiments, budgets, queue depth and the kill switch.
  //
  // The queue panel shows **reserved cost** beside depth, because jobs differ by
  // orders of magnitude in spend and depth alone tells an operator nothing about
  // what is about to be spent.
  import Fetch from '../lib/Fetch.svelte'
  import {
    experiments,
    jobs,
    budgets,
    killswitches,
    pauseExperiment,
    resumeExperiment,
    cancelExperiment,
    engage,
    release,
    money,
    ms,
    short,
  } from '../lib/api.js'

  let outcome = null
  let reason = ''

  async function act(fn, refresh) {
    outcome = null
    try {
      await fn()
      await refresh()
      outcome = { ok: true, text: 'done' }
    } catch (error) {
      outcome = { ok: false, text: error.message }
    }
  }

  async function everything() {
    const [e, j, b, k] = await Promise.all([
      experiments(),
      jobs(),
      budgets(),
      killswitches(),
    ])
    return { experiments: e.experiments, shards: j.shards, budgets: b, kill: k }
  }
</script>

<h2>Operations</h2>

{#if outcome}
  <p class={outcome.ok ? 'ok' : 'error'}>{outcome.text}</p>
{/if}

<Fetch load={everything} every={8000} let:data let:refresh>
  <h3>Kill switch</h3>
  {#if data.kill.live.length}
    <ul>
      {#each data.kill.live as engagement}
        <li class="engaged">
          <strong>{JSON.stringify(engagement.scope)}</strong>
          — {engagement.mode}, {engagement.reason_code}, by {engagement.engaged_by}
          {#if engagement.entry != null}
            <span class="muted">
              (revocation entry {engagement.entry}; each deployment picks it up from
              the feed it already pulls)
            </span>
          {/if}
          <button on:click={() => act(() => release(engagement.id), refresh)}>
            release
          </button>
        </li>
      {/each}
    </ul>
  {:else}
    <p class="muted">Nothing is engaged.</p>
  {/if}
  <div class="actions">
    <input placeholder="reason" bind:value={reason} />
    <button
      class="danger"
      on:click={() => act(() => engage({ scope: 'global' }, 'drain', reason), refresh)}
      >Drain everything</button
    >
    <button
      class="danger"
      on:click={() => act(() => engage({ scope: 'global' }, 'stop', reason), refresh)}
      >Stop everything</button
    >
  </div>
  <p class="muted">
    Engaging is an operator's act and releasing is an admin's: stopping the system
    is urgent and reversible, restarting it is a decision.
  </p>

  <h3>Experiments</h3>
  {#if !data.experiments.length}
    <p class="muted">None submitted.</p>
  {:else}
    <table>
      <thead>
        <tr><th>id</th><th>lineage</th><th>state</th><th>shard</th><th></th></tr>
      </thead>
      <tbody>
        {#each data.experiments as experiment}
          <tr>
            <td><code>{experiment.id}</code></td>
            <td><code title={experiment.lineage_root}>{short(experiment.lineage_root)}</code></td>
            <td>{experiment.state}{#if experiment.detail} — {experiment.detail}{/if}</td>
            <td><code>{experiment.shard}</code></td>
            <td>
              {#if experiment.state === 'paused'}
                <button on:click={() => act(() => resumeExperiment(experiment.id), refresh)}>
                  resume
                </button>
              {:else if experiment.state === 'running' || experiment.state === 'queued'}
                <button on:click={() => act(() => pauseExperiment(experiment.id), refresh)}>
                  pause
                </button>
                <button
                  on:click={() =>
                    act(() => cancelExperiment(experiment.id, reason || 'operator'), refresh)}
                >
                  cancel
                </button>
              {/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}

  <h3>Queue</h3>
  <table>
    <thead>
      <tr>
        <th>shard</th>
        <th>ready</th>
        <th>in flight</th>
        <th>oldest ready</th>
        <th>reserved cost</th>
        <th>dead letter</th>
      </tr>
    </thead>
    <tbody>
      {#each Object.entries(data.shards) as [shard, depth]}
        <tr>
          <td><code>{shard}</code></td>
          <td>{depth.ready}</td>
          <td>{depth.in_flight}</td>
          <td>{ms(depth.oldest_ready_ms)}</td>
          <td>{money(depth.reserved_cost)}</td>
          <td class:bad={depth.dead_letter > 0}>{depth.dead_letter}</td>
        </tr>
      {/each}
    </tbody>
  </table>

  <h3>Budgets</h3>
  {#if data.budgets.alerts.length}
    <p class="warn">soft cap crossed: {data.budgets.alerts.join(', ')}</p>
  {/if}
  {#if !data.budgets.entries.length}
    <p class="muted">Nothing spent yet.</p>
  {:else}
    <table>
      <thead>
        <tr><th>budget</th><th>run</th><th>cost</th><th>tokens</th></tr>
      </thead>
      <tbody>
        {#each data.budgets.entries.slice(-25).reverse() as entry}
          <tr>
            <td><code>{entry.budget_id}</code></td>
            <td><code>{entry.run_id}</code></td>
            <td>{money(entry.spent.cost)}</td>
            <td>{entry.spent.tokens}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</Fetch>

<style>
  .engaged {
    color: var(--bad);
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    margin: 0.5rem 0;
  }
  .danger {
    border-color: var(--bad);
    color: var(--bad);
  }
  .bad {
    color: var(--bad);
    font-weight: 600;
  }
  .warn {
    color: var(--warn);
  }
</style>
