<script>
  // Allocation: shares, states and the capacity *bounds*.
  //
  // Saving produces a new signed DeploymentPolicy that each instance pulls on its
  // next feed interval. This page therefore shows three things a console that
  // called production would not need: the diff against what is in force, who signed
  // it, and which instances have actually picked it up.
  //
  // The capacity controller itself is not editable here. It is a closed loop inside
  // each deployment; what a human sets are its bounds, and an operator fighting a
  // controller at a 30-second round trip would lose.
  import Fetch from '../lib/Fetch.svelte'
  import { policy, issuePolicy, short, pct } from '../lib/api.js'

  let draft = null
  let outcome = null
  let busy = false

  // The defaults bundle.md fixes (contracts: CapacityPolicy::default). A fresh
  // deployment has issued no policy, so there is nothing to copy from and the form
  // still has to render: without these the nested `slo` is undefined and the first
  // binding throws before anything is drawn.
  const DEFAULT_CAPACITY = {
    slo: { metric: 'ttft_ms', target_p99: 2000 },
    front_reserve: 0.8,
    internal_max: 0.2,
    internal_min: 0.0,
    shadow_share: 0.05,
    preempt_grace_ms: 1000,
  }

  function start(data) {
    if (draft) return draft
    const source = data.draft ?? data.in_force ?? {}
    const capacity = source.capacity ?? data.in_force?.capacity ?? {}
    draft = {
      deployments: (source.deployments ?? []).map((d) => ({ ...d })),
      // Deep, not shallow: a spread shares the nested `slo` with the object the
      // poll replaces every 20 s, so editing the SLO target would mutate the
      // fetched read model and the next poll would discard the edit.
      capacity: {
        ...DEFAULT_CAPACITY,
        ...capacity,
        slo: { ...DEFAULT_CAPACITY.slo, ...(capacity.slo ?? {}) },
      },
    }
    return draft
  }

  async function save(refresh) {
    busy = true
    outcome = null
    try {
      const result = await issuePolicy(draft.deployments, draft.capacity)
      outcome = {
        ok: true,
        text: `signed version ${result.policy.version} with key ${result.policy.signature.key_id}; ${result.delivery}`,
      }
      draft = null
      await refresh()
    } catch (error) {
      outcome = { ok: false, text: error.message }
    } finally {
      busy = false
    }
  }
</script>

<h2>Allocation</h2>

{#if outcome}
  <p class={outcome.ok ? 'ok' : 'error'}>{outcome.text}</p>
{/if}

<Fetch load={policy} every={20000} let:data let:refresh>
  <!-- `start(data)` fills the module-level `draft` before the body renders. The
       body binds to `draft` rather than to a local: Svelte cannot bind to a
       variable declared with {@const}, which is why this page never compiled. -->
  {#if start(data)}

  <h3>In force</h3>
  {#if data.in_force}
    <p>
      version <strong>{data.in_force.version}</strong>, signed by
      <code>{data.in_force.signature.key_id}</code>, policy
      <code title={data.in_force.policy_id}>{short(data.in_force.policy_id)}</code>
    </p>
    {#if data.adoption}
      <p>
        picked up by: {data.adoption.adopted_by.join(', ') || 'no instance yet'}
        {#if data.adoption.pending.length}
          <span class="warn">
            — still on an older version: {data.adoption.pending.join(', ')}
          </span>
        {/if}
      </p>
    {/if}
  {:else}
    <p class="muted">No policy has been issued. Deployments run on local config.</p>
  {/if}

  <h3>Draft</h3>
  <table>
    <thead>
      <tr><th>deployment</th><th>state</th><th>share</th></tr>
    </thead>
    <tbody>
      {#each draft.deployments as deployment}
        <tr>
          <td><code title={deployment.root}>{short(deployment.root)}</code></td>
          <td>
            <select bind:value={deployment.state}>
              {#each ['shadow', 'canary', 'active', 'draining', 'retired', 'quarantined'] as state}
                <option value={state}>{state}</option>
              {/each}
            </select>
          </td>
          <td>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              bind:value={deployment.share}
            />
            <span class="muted">{pct(deployment.share)}</span>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>

  <h3>Capacity bounds</h3>
  <p class="muted">
    The controller's bounds, not its decisions: it is local to each deployment and
    reacts immediately, which is why this page cannot set the share per request.
  </p>
  <div class="bounds">
    <label>
      SLO target p99 (ms)
      <input type="number" min="1" bind:value={draft.capacity.slo.target_p99} />
    </label>
    <label>
      front-facing reserve
      <input type="number" min="0" max="1" step="0.05" bind:value={draft.capacity.front_reserve} />
    </label>
    <label>
      internal ceiling
      <input type="number" min="0" max="1" step="0.05" bind:value={draft.capacity.internal_max} />
    </label>
    <label>
      internal floor
      <input type="number" min="0" max="1" step="0.05" bind:value={draft.capacity.internal_min} />
    </label>
    <label>
      shadow share
      <input type="number" min="0" max="1" step="0.01" bind:value={draft.capacity.shadow_share} />
    </label>
    <label>
      preempt grace (ms)
      <input type="number" min="0" bind:value={draft.capacity.preempt_grace_ms} />
    </label>
  </div>

  {#if data.diff && data.diff.length}
    <h3>Diff against what is in force</h3>
    <ul class="diff">
      {#each data.diff as change}
        <li><code>{JSON.stringify(change)}</code></li>
      {/each}
    </ul>
  {/if}

    <button disabled={busy} on:click={() => save(refresh)}>Sign and publish</button>
    <button class="ghost" on:click={() => (draft = null)}>Discard draft</button>
  {/if}
</Fetch>

<style>
  .bounds {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
  }
  .bounds label {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .diff li {
    font-size: 0.9em;
  }
  .warn {
    color: var(--warn);
  }
  .ghost {
    background: none;
  }
</style>
