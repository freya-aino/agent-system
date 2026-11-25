<script>
  // Live swarms and their members.
  //
  // The internal cohort is shown as what it is: locked candidates being measured on
  // the live system, with the capacity they get and how often they were preempted.
  // Mixing them into the member list would make a page that implies clients are
  // reaching them.
  import Fetch from '../lib/Fetch.svelte'
  import Distribution from '../lib/Distribution.svelte'
  import { deployments, members, short, pct } from '../lib/api.js'

  let selected = null

  async function swarm() {
    const { deployments: list } = await deployments()
    if (!list.length) return { list: [], front: [], internal: [] }
    const root = selected ?? list[0].bundle
    selected = root
    const { members: all } = await members(root)
    return {
      list,
      root,
      front: all.filter((m) => m.exposure === 'front'),
      internal: all.filter((m) => m.exposure !== 'front'),
    }
  }
</script>

<h2>Swarms and members</h2>

<Fetch load={swarm} every={10000} let:data>
  {#if data.list.length > 1}
    <label>
      deployment
      <select bind:value={selected}>
        {#each data.list as deployment}
          <option value={deployment.bundle}>{short(deployment.bundle)}</option>
        {/each}
      </select>
    </label>
  {/if}

  <h3>Serving</h3>
  {#if !data.front.length}
    <p class="muted">No member is serving traffic in this deployment.</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>member</th>
          <th>slot</th>
          <th>share</th>
          <th>requests</th>
          <th>preempted</th>
          <th>live fitness</th>
        </tr>
      </thead>
      <tbody>
        {#each data.front as member}
          <tr>
            <td><code title={member.member}>{short(member.member)}</code></td>
            <td>{member.slot || '—'}</td>
            <td>{pct(member.share)}</td>
            <td>
              {#each Object.entries(member.requests_by_class) as [cls, count]}
                <span class="chip">{cls}: {count}</span>
              {/each}
            </td>
            <td>{member.preempted}</td>
            <td>
              {#each member.fitness as record}
                <div>
                  <code title={record.metric}>{short(record.metric)}</code>
                  <Distribution distribution={record.distribution} n={record.n} />
                </div>
              {/each}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}

  <h3>Internal cohort</h3>
  <p class="muted">
    Locked candidates measured on the live system. They serve no client, nothing
    they produce is stored, and they are preempted whenever front-facing traffic
    needs the capacity.
  </p>
  {#if !data.internal.length}
    <p class="muted">No candidate is being measured on this deployment.</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>candidate</th>
          <th>capacity</th>
          <th>runs</th>
          <th>preempted</th>
          <th>live fitness</th>
        </tr>
      </thead>
      <tbody>
        {#each data.internal as member}
          <tr>
            <td><code title={member.member}>{short(member.member)}</code></td>
            <td>{pct(member.share)}</td>
            <td>
              {#each Object.entries(member.requests_by_class) as [cls, count]}
                <span class="chip">{cls}: {count}</span>
              {/each}
            </td>
            <td>{member.preempted}</td>
            <td>
              {#each member.fitness as record}
                <div>
                  <code title={record.metric}>{short(record.metric)}</code>
                  <Distribution distribution={record.distribution} n={record.n} />
                </div>
              {/each}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</Fetch>

<style>
  .chip {
    display: inline-block;
    background: var(--chip);
    border-radius: 3px;
    padding: 0 0.35em;
    margin-right: 0.3em;
    font-size: 0.85em;
  }
</style>
