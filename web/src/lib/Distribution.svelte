<script>
  // A distribution, **never a bare number**.
  //
  // A mean over three runs and a mean over three hundred are different claims, so
  // `n` is not optional here: the component refuses to render without it rather
  // than drawing a confident-looking bar over nothing. That refusal is the point —
  // it makes an under-evidenced comparison impossible to show by accident.
  export let distribution = null
  export let n = null
  export let label = ''

  $: count = n ?? distribution?.n ?? null
  $: ci = distribution?.bootstrap_ci
  $: mean = distribution?.mean
</script>

{#if count == null || mean == null}
  <span class="missing" title="a figure with no n is not a measurement">
    no evidence
  </span>
{:else}
  <span class="dist" title={label}>
    <strong>{mean.toFixed(3)}</strong>
    {#if ci}
      <span class="ci">[{ci.lo.toFixed(3)}, {ci.hi.toFixed(3)}]</span>
    {/if}
    <span class="n" class:thin={count < 20}>n={count}</span>
  </span>
{/if}

<style>
  .dist {
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .ci {
    color: var(--muted);
    margin-left: 0.3em;
  }
  .n {
    color: var(--muted);
    margin-left: 0.4em;
    font-size: 0.85em;
  }
  /* Few replicates are marked, because the reader's confidence should track the
     evidence rather than the rendering. */
  .thin {
    color: var(--warn);
    font-weight: 600;
  }
  .missing {
    color: var(--muted);
    font-style: italic;
  }
</style>
