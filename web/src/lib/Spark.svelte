<script>
  // A series as inline SVG. Twenty lines against a charting dependency.
  /** @type {(number | null)[]} */
  export let points = []
  /** @type {number | null} */
  export let target = null
  export let height = 40
  export let width = 240
  export let unit = ''

  $: values = points.filter((v) => typeof v === 'number')
  $: max = Math.max(...values, target ?? 0, 1)
  $: path = values
    .map((value, index) => {
      const x = values.length > 1 ? (index / (values.length - 1)) * width : 0
      const y = height - (value / max) * height
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  $: targetY = target == null ? null : height - (target / max) * height
  $: last = values.at(-1)
  $: breaching = target != null && last != null && last > target
</script>

<span class="spark">
  <svg {width} {height} role="img" aria-label="series">
    {#if targetY != null}
      <!-- The target is drawn, not described: the question is always whether the
           line is under it. -->
      <line x1="0" y1={targetY} x2={width} y2={targetY} class="target" />
    {/if}
    <path d={path} class:breaching />
  </svg>
  <span class="last" class:breaching>{last == null ? '—' : last.toFixed(0)}{unit}</span>
</span>

<style>
  .spark {
    display: inline-flex;
    align-items: center;
    gap: 0.5em;
  }
  path {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.5;
  }
  path.breaching {
    stroke: var(--bad);
  }
  .target {
    stroke: var(--muted);
    stroke-dasharray: 3 3;
    stroke-width: 1;
  }
  .last {
    font-variant-numeric: tabular-nums;
    color: var(--muted);
  }
  .last.breaching {
    color: var(--bad);
    font-weight: 600;
  }
</style>
