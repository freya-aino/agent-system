//! The event feed: a paged query and an SSE subscription.
//!
//! **The log is the source of truth and this is not it.** A real deployment serves
//! these from the store (C4's `query` and `subscribe`), which is where the history
//! and the hash chain live. What this module adds is the *recent* window every
//! dashboard needs and no store query serves cheaply: a bounded ring the API can
//! read without a round trip, and a cursor a browser can resume from.
//!
//! **Bounded, and it says so.** The ring holds the last [`WINDOW`] events; a cursor
//! that has fallen behind is told it has, rather than being served a silently
//! truncated range. A feed that quietly skipped events would make the one thing a
//! dashboard is for — noticing — unreliable.
//!
//! **Content-free, like everything else here.** What flows through is kinds,
//! subjects, sequences and the small `data` objects the event kinds define. There is
//! no payload to leak because S1 never receives one.

use std::collections::VecDeque;

use agent_contracts::store::EventFilter;
use agent_contracts::trace_event::{Actor, Event};
use serde_json::{Value, json};

/// How many recent events the ring holds per node. A few minutes of a busy shard,
/// which is what a dashboard window is.
pub const WINDOW: usize = 1_000;

/// One line as the feed serves it. The contract `Event` without its chain fields:
/// `prev` and the hash chain are the store's business, and a dashboard that showed
/// them would imply it had verified them.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Line {
    pub seq: u64,
    pub shard: String,
    pub kind: String,
    pub actor: String,
    pub subject: String,
    pub ts: i64,
    pub data: Value,
}

impl From<&Event> for Line {
    fn from(event: &Event) -> Line {
        Line {
            seq: event.seq,
            shard: event.shard.clone(),
            kind: event.kind.clone(),
            actor: match &event.actor {
                Actor::Human(id) => format!("human:{id}"),
                Actor::Optimizer(id) => format!("optimizer:{id}"),
                Actor::System(id) => format!("system:{id}"),
                Actor::Service(id) => format!("service:{id}"),
            },
            subject: event.subject.as_str().to_owned(),
            ts: event.ts,
            data: event.data.clone(),
        }
    }
}

/// The bounded ring.
#[derive(Debug, Default)]
pub struct Feed {
    lines: VecDeque<Line>,
    /// The monotonic cursor the feed hands out. Not a shard sequence: a feed spans
    /// shards, and a browser needs one number to resume from.
    next_cursor: u64,
    /// The cursor of the oldest line still held, so a caller that has fallen behind
    /// can be told rather than quietly served a gap.
    oldest_cursor: u64,
    cursors: VecDeque<u64>,
}

/// One page of the feed.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Page {
    pub lines: Vec<Line>,
    /// Pass this as `since` to continue.
    pub cursor: u64,
    /// True when the caller's `since` was older than the ring: events between were
    /// evicted and must be read from the store.
    pub lagged: bool,
    /// The oldest cursor still available here.
    pub oldest: u64,
}

impl Feed {
    pub fn new() -> Feed {
        Feed::default()
    }

    /// Record an event. Called by the writer after it appends, never before: the
    /// feed shows what is in the log, not what is about to be.
    pub fn record(&mut self, event: &Event) {
        self.push(Line::from(event));
    }

    /// Record a line the writer produced from a draft. The same rule applies.
    pub fn record_line(&mut self, line: Line) {
        self.push(line);
    }

    fn push(&mut self, line: Line) {
        self.next_cursor += 1;
        self.lines.push_back(line);
        self.cursors.push_back(self.next_cursor);
        while self.lines.len() > WINDOW {
            self.lines.pop_front();
            if let Some(evicted) = self.cursors.pop_front() {
                self.oldest_cursor = evicted;
            }
        }
    }

    /// A page from `since`, filtered.
    pub fn query(&self, since: u64, filter: &EventFilter, limit: usize) -> Page {
        let lagged = since > 0 && since < self.oldest_cursor;
        let mut lines = vec![];
        for (cursor, line) in self.cursors.iter().zip(self.lines.iter()) {
            if *cursor <= since {
                continue;
            }
            if !matches(line, filter) {
                continue;
            }
            lines.push(line.clone());
            if lines.len() >= limit {
                break;
            }
        }
        Page {
            lines,
            cursor: self.next_cursor,
            lagged,
            oldest: self.oldest_cursor,
        }
    }

    pub fn cursor(&self) -> u64 {
        self.next_cursor
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// Whether a line matches a filter. Only the predicates a dashboard uses: kind,
/// subject, actor and shard. A filter this feed cannot honour is a query for the
/// store, and [`unsupported`] says which.
pub fn matches(line: &Line, filter: &EventFilter) -> bool {
    if let Some(kinds) = &filter.kind
        && !kinds.iter().any(|k| k == &line.kind)
    {
        return false;
    }
    if let Some(subjects) = &filter.subject
        && !subjects.iter().any(|s| s.as_str() == line.subject)
    {
        return false;
    }
    if let Some(actors) = &filter.actor
        && !actors.iter().any(|a| a == &line.actor)
    {
        return false;
    }
    if let Some(experiment) = &filter.experiment
        && line.data.get("experiment").and_then(Value::as_str) != Some(experiment.as_str())
    {
        return false;
    }
    true
}

/// The filter fields this feed cannot answer, named rather than ignored.
///
/// `lineage_of` expands to a genealogy closure and `data` predicates index the
/// store: both are reads the store does properly, and answering them approximately
/// here would give a dashboard a subtly wrong set.
pub fn unsupported(filter: &EventFilter) -> Vec<&'static str> {
    let mut missing = vec![];
    if filter.lineage_of.is_some() {
        missing.push("lineage_of");
    }
    if filter.data.is_some() {
        missing.push("data");
    }
    if filter.seq.is_some() {
        missing.push("seq");
    }
    if filter.ts.is_some() {
        missing.push("ts");
    }
    missing
}

/// Parse a query string into a filter. Only what the feed supports; anything else
/// is reported rather than silently dropped.
pub fn filter_from_query(query: &str) -> (EventFilter, Vec<String>) {
    let mut filter = EventFilter {
        kind: None,
        subject: None,
        actor: None,
        seq: None,
        ts: None,
        data: None,
        lineage_of: None,
        experiment: None,
    };
    let mut ignored = vec![];
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        let value = value.replace("%3A", ":").replace("%2C", ",");
        match key {
            "kind" => {
                filter.kind = Some(value.split(',').map(str::to_owned).collect());
            }
            "actor" => {
                filter.actor = Some(value.split(',').map(str::to_owned).collect());
            }
            "subject" => {
                let subjects: Vec<agent_contracts::types::HashId> = value
                    .split(',')
                    .filter_map(|s| agent_contracts::types::HashId::parse(s).ok())
                    .collect();
                filter.subject = Some(subjects);
            }
            "experiment" => filter.experiment = Some(value),
            "since" | "limit" => {}
            other => ignored.push(other.to_owned()),
        }
    }
    (filter, ignored)
}

/// One SSE frame. Written out here because the format is three lines and a crate
/// for it would be a dependency for a `format!`.
pub fn sse_frame(line: &Line, cursor: u64) -> String {
    let data = serde_json::to_string(line).unwrap_or_else(|_| "{}".to_owned());
    format!("id: {cursor}\nevent: {}\ndata: {data}\n\n", line.kind)
}

/// The frame that tells a client it has fallen behind the ring. A comment frame
/// would be silently dropped by `EventSource`, so this is a real event the UI can
/// act on.
pub fn sse_lagged(oldest: u64) -> String {
    format!(
        "event: lagged\ndata: {}\n\n",
        json!({"oldest": oldest, "detail": "read the store for what came before"})
    )
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::{EventId, HashId};

    use super::*;

    fn event(kind: &str, seq: u64, subject: u8, data: Value) -> Event {
        Event {
            id: EventId {
                shard: "pop-1:0".into(),
                seq,
            },
            shard: "pop-1:0".into(),
            seq,
            prev: HashId::from_sha256([0; 32]),
            ts: 1_000 + seq as i64,
            actor: Actor::System("s1".into()),
            kind: kind.to_owned(),
            subject: HashId::from_sha256([subject; 32]),
            data,
            evidence: vec![],
        }
    }

    fn feed_with(count: u64) -> Feed {
        let mut feed = Feed::new();
        for seq in 1..=count {
            feed.record(&event("committed", seq, 1, json!({})));
        }
        feed
    }

    #[test]
    fn a_page_resumes_from_its_cursor_without_repeating() {
        let feed = feed_with(5);
        let first = feed.query(0, &filter_from_query("").0, 3);
        assert_eq!(first.lines.len(), 3);
        assert!(!first.lagged);
        let second = feed.query(3, &filter_from_query("").0, 10);
        assert_eq!(second.lines.len(), 2);
        assert_eq!(second.lines[0].seq, 4, "nothing is served twice");
        assert_eq!(second.cursor, 5);
        assert!(feed.query(5, &filter_from_query("").0, 10).lines.is_empty());
    }

    #[test]
    fn a_caller_that_fell_behind_the_ring_is_told_rather_than_served_a_gap() {
        let feed = feed_with(WINDOW as u64 + 50);
        assert_eq!(feed.len(), WINDOW);
        let page = feed.query(1, &filter_from_query("").0, 10);
        assert!(
            page.lagged,
            "a feed that quietly skipped events would make noticing unreliable"
        );
        assert!(page.oldest > 1);
        // And a current cursor is not lagged.
        assert!(
            !feed
                .query(feed.cursor() - 1, &filter_from_query("").0, 10)
                .lagged
        );
    }

    #[test]
    fn the_filters_a_dashboard_uses_are_honoured() {
        let mut feed = Feed::new();
        feed.record(&event("committed", 1, 1, json!({"experiment": "exp-1"})));
        feed.record(&event("retired", 2, 2, json!({"experiment": "exp-2"})));
        feed.record(&event("committed", 3, 2, json!({"experiment": "exp-2"})));

        let (kinds, _) = filter_from_query("kind=committed");
        assert_eq!(feed.query(0, &kinds, 10).lines.len(), 2);

        let (subject, _) = filter_from_query(&format!(
            "subject={}",
            HashId::from_sha256([2; 32]).as_str()
        ));
        assert_eq!(feed.query(0, &subject, 10).lines.len(), 2);

        let (experiment, _) = filter_from_query("experiment=exp-2");
        assert_eq!(feed.query(0, &experiment, 10).lines.len(), 2);

        let (actor, _) = filter_from_query("actor=system:s1");
        assert_eq!(feed.query(0, &actor, 10).lines.len(), 3);
        let (nobody, _) = filter_from_query("actor=human:ada");
        assert!(feed.query(0, &nobody, 10).lines.is_empty());
    }

    #[test]
    fn a_filter_this_feed_cannot_answer_is_named_rather_than_ignored() {
        let mut filter = filter_from_query("").0;
        filter.lineage_of = Some(HashId::from_sha256([1; 32]));
        filter.data = Some(vec![]);
        let missing = unsupported(&filter);
        assert!(missing.contains(&"lineage_of"));
        assert!(
            missing.contains(&"data"),
            "answering it approximately would give a subtly wrong set"
        );

        // And an unknown query parameter is reported too.
        let (_, ignored) = filter_from_query("kind=committed&weird=1");
        assert_eq!(ignored, vec!["weird".to_owned()]);
    }

    #[test]
    fn a_line_carries_no_chain_fields() {
        let mut feed = Feed::new();
        feed.record(&event("committed", 1, 1, json!({"x": 1})));
        let line = &feed.query(0, &filter_from_query("").0, 1).lines[0];
        let encoded = serde_json::to_value(line).unwrap();
        assert!(
            encoded.get("prev").is_none() && encoded.get("evidence").is_none(),
            "a dashboard showing the chain would imply it had verified it"
        );
        assert_eq!(encoded["kind"], "committed");
        assert_eq!(encoded["actor"], "system:s1");
    }

    #[test]
    fn an_sse_frame_carries_its_cursor_so_a_browser_can_resume() {
        let mut feed = Feed::new();
        feed.record(&event("approved", 1, 1, json!({})));
        let line = feed.query(0, &filter_from_query("").0, 1).lines[0].clone();
        let frame = sse_frame(&line, 7);
        assert!(frame.starts_with("id: 7\n"));
        assert!(frame.contains("event: approved\n"));
        assert!(frame.ends_with("\n\n"), "a frame ends with a blank line");
        assert!(sse_lagged(42).contains("event: lagged"));
    }
}
