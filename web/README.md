# The management interface

Svelte + Vite, served by the S1 binary as static assets. S2 has no UI.

```sh
npm install          # exact versions; see DEPS.md
npm run dev          # proxies to S1 on 127.0.0.1:8080
npm run build        # emits dist/
```

Then point S1 at the build:

```sh
AGENT_S1_WEB_ROOT=$PWD/dist cargo run -p agent-s1 -- --dev /tmp/s1
```

Until it is built, S1 serves a page saying so rather than a blank 404.

## What the views are for

| View | What it answers |
|---|---|
| Served surface | What can the outside world actually call, per protocol, against which schemas? |
| Swarms | Which members are serving, which candidates are being measured on the live system, and what is each one's fitness **with `n`**? |
| Promotions | What is waiting to be incorporated, on what evidence (live versus inbound), and what is still blocking it? |
| Allocation | What shares and bounds are in force, what would change, who signed it, and which instances have picked it up? |
| Capacity | Is front-facing p99 under the target, how much capacity is internal work getting, and **why** did the controller settle there? |
| Operations | Experiments, queue depth with reserved cost, budget spend, and the kill switch. |

## Rules this UI follows

- **No permission is guessed.** Controls are not hidden by role; the request is
  sent and the service's refusal is shown, because the refusal names the role it
  would need.
- **No bare numbers.** `Distribution.svelte` refuses to render a figure without
  `n`, and marks a thin one.
- **No payload view.** The event strip shows kinds and sequences. A `no_sample`
  run has no payload to show — for any role, including admin.
- **Nothing calls S2.** Every action here changes S1 state or produces a signed
  artifact a deployment pulls.
