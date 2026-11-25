<script>
  // The evaluation-versus-latency trade, made visible rather than inferred.
  //
  // Front-facing p99 against the target, the internal share the controller settled
  // on **and why**, preemptions, and sheds. Without the reason the series is a
  // mystery: a low internal share because the SLO is tight and a low one because
  // the ceiling is binding call for opposite actions.
  import Fetch from '../lib/Fetch.svelte'
  import Spark from '../lib/Spark.svelte'
  import { deployments, short, pct } from '../lib/api.js'
</script>

<h2>Capacity and latency</h2>

<Fetch load={deployments} every={10000} let:data>
  {#if !data.deployments.length}
    <p class="muted">No deployment is reporting.</p>
  {:else}
    {#each data.deployments as deployment}
      {@const series = deployment.capacity ?? []}
      {@const latest = series.at(-1)}
      <section>
        <h3><code title={deployment.bundle}>{short(deployment.bundle)}</code></h3>
        <p class="muted">
          instances: {deployment.instances.join(', ') || 'none'} · last window
          {deployment.last_seq}
        </p>
        {#if !latest}
          <p class="muted">No capacity report yet.</p>
        {:else}
          <table>
            <tbody>
              <tr>
                <th>front-facing p99</th>
                <td>
                  <Spark
                    points={series.map((s) => s.observed_p99)}
                    target={latest.target_p99}
                    unit="ms"
                  />
                  <span class="muted">target {latest.target_p99}ms</span>
                </td>
              </tr>
              <tr>
                <th>internal share</th>
                <td>
                  <Spark points={series.map((s) => s.internal_share * 100)} unit="%" />
                  <span class="muted">
                    now {pct(latest.internal_share)} — because
                    <strong>{latest.reason}</strong>
                  </span>
                </td>
              </tr>
              <tr>
                <th>preempted</th>
                <td><Spark points={series.map((s) => s.preempted)} /></td>
              </tr>
              <tr>
                <th>shed</th>
                <td>
                  <Spark points={series.map((s) => s.shed)} />
                  <span class="muted">
                    {latest.shed > 0
                      ? 'front-facing demand alone exceeded capacity'
                      : 'nothing shed'}
                  </span>
                </td>
              </tr>
              <tr>
                <th>SLO breaches</th>
                <td><Spark points={series.map((s) => s.breaches)} /></td>
              </tr>
            </tbody>
          </table>
        {/if}
      </section>
    {/each}
  {/if}
</Fetch>

<style>
  section {
    border-top: 1px solid var(--line);
    padding-top: 0.75rem;
    margin-top: 1rem;
  }
  th {
    text-align: left;
    font-weight: 500;
    color: var(--muted);
    padding-right: 1rem;
  }
</style>
