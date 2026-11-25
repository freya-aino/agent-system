<script>
  // The incorporate workflow: the main thing this interface is for.
  //
  // A candidate is shown with its gate evidence split **live versus inbound**,
  // because a gate judged on production evidence and one judged on inbound
  // evaluation are different claims. Approve is disabled only when the *service*
  // says the proposal is blocked — never because of a role guess, since the refusal
  // is more informative than a greyed-out button.
  import Fetch from '../lib/Fetch.svelte'
  import Distribution from '../lib/Distribution.svelte'
  import { promotions, approve, reject, short } from '../lib/api.js'

  let note = ''
  let busy = null
  let outcome = null

  async function decide(id, which, refresh) {
    busy = id
    outcome = null
    try {
      const result = which === 'approve' ? await approve(id, note) : await reject(id, note)
      outcome = {
        ok: true,
        text:
          which === 'approve'
            ? result.build
              ? 'approved; a build has been queued'
              : `approved; still blocked by: ${(result.blocked_by || []).join(', ')}`
            : 'rejected',
      }
      note = ''
      await refresh()
    } catch (error) {
      outcome = { ok: false, text: error.message }
    } finally {
      busy = null
    }
  }
</script>

<h2>Promotion queue</h2>
<p class="muted">
  Approving writes <code>approved{'{'}actor, evidence{'}'}</code> and triggers a
  build. That is how a proposed change reaches production; nothing here calls a
  deployment.
</p>

{#if outcome}
  <p class={outcome.ok ? 'ok' : 'error'}>{outcome.text}</p>
{/if}

<Fetch load={promotions} every={10000} let:data let:refresh>
  {#if !data.pending.length}
    <p class="muted">Nothing is waiting. The search keeps running regardless.</p>
  {:else}
    {#each data.pending as entry}
      <section class="proposal">
        <header>
          <h3><code title={entry.candidate}>{short(entry.candidate)}</code></h3>
          <span class="chip">{entry.class}</span>
          <span class="muted">proposal <code>{short(entry.proposal)}</code></span>
        </header>

        <dl>
          <dt>gates on inbound evaluation</dt>
          <dd>{entry.inbound_gates}</dd>
          <dt>gates on live evidence</dt>
          <dd>{entry.live_gates}</dd>
          <dt>decisions recorded</dt>
          <dd>{entry.decisions}</dd>
        </dl>

        {#if entry.effect}
          <p>
            effect against the incumbent:
            <Distribution distribution={entry.effect} label="paired effect size" />
          </p>
        {/if}

        {#if entry.blocked_by && entry.blocked_by.length}
          <p class="blocked">
            blocked by: {entry.blocked_by.join(', ')}
          </p>
        {/if}

        <div class="actions">
          <input placeholder="note (recorded with the decision)" bind:value={note} />
          <button
            disabled={busy === entry.proposal}
            on:click={() => decide(entry.proposal, 'approve', refresh)}>Approve</button
          >
          <button
            class="reject"
            disabled={busy === entry.proposal}
            on:click={() => decide(entry.proposal, 'reject', refresh)}>Reject</button
          >
        </div>
      </section>
    {/each}
  {/if}
</Fetch>

<style>
  .proposal {
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
  }
  header h3 {
    margin: 0;
  }
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.2rem 1rem;
    margin: 0.75rem 0;
  }
  dt {
    color: var(--muted);
  }
  .blocked {
    color: var(--warn);
  }
  .actions {
    display: flex;
    gap: 0.5rem;
  }
  .actions input {
    flex: 1;
  }
  .reject {
    border-color: var(--bad);
    color: var(--bad);
  }
  .chip {
    background: var(--chip);
    border-radius: 3px;
    padding: 0 0.35em;
  }
</style>
