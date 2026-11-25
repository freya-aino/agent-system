<script>
  // The served surface: the deployment's public face.
  //
  // What the outside world can actually call, per protocol. This is the page a
  // caller's integration is checked against, so the schemas shown here are the ones
  // the front validates — not a copy maintained beside them.
  import Fetch from '../lib/Fetch.svelte'
  import { deployments, functions, short, pct } from '../lib/api.js'

  let selected = null

  async function surface() {
    const { deployments: list } = await deployments()
    if (!list.length) return { rows: [] }
    const root = selected ?? list[0].bundle
    selected = root
    const { functions: fns } = await functions(root)
    return { list, root, rows: fns }
  }
</script>

<h2>Served surface</h2>
<p class="muted">
  Every advertised function, as a caller reaches it. The input and output schemas
  here are the ones the front validates against.
</p>

<Fetch load={surface} every={15000} let:data>
  {#if data.list && data.list.length > 1}
    <label>
      deployment
      <select bind:value={selected}>
        {#each data.list as deployment}
          <option value={deployment.bundle}>{short(deployment.bundle)}</option>
        {/each}
      </select>
    </label>
  {/if}

  {#if !data.rows.length}
    <p class="muted">
      No bundle is advertising anything. A deployment appears here once it is
      active and its surface has been registered.
    </p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>protocol</th>
          <th>advertised as</th>
          <th>endpoint</th>
          <th>scope</th>
          <th>timeout</th>
          <th>in → out</th>
          <th>adapter</th>
          <th>auth</th>
          <th>quota</th>
          <th>isolation</th>
          <th>state</th>
          <th>share</th>
        </tr>
      </thead>
      <tbody>
        {#each data.rows as fn}
          <tr>
            <td><code>{fn.protocol}</code></td>
            <td><strong>{fn.advertised_as}</strong></td>
            <td><code>{fn.endpoint}</code></td>
            <td>{fn.descriptor.scope}</td>
            <td>{fn.descriptor.timeout_ms ?? '—'}</td>
            <td>
              <code title={fn.descriptor.input}>{short(fn.descriptor.input)}</code>
              →
              <code title={fn.descriptor.output}>{short(fn.descriptor.output)}</code>
            </td>
            <td>{fn.adapter}@{fn.adapter_version}</td>
            <td>{fn.auth_mode}</td>
            <td>{fn.quota_per_caller ?? 'none'}</td>
            <td>{fn.isolation_level}</td>
            <td>{fn.deploy_state}</td>
            <td>{pct(fn.traffic_share)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</Fetch>
