-- C4 store schema (crates/store/specs/C4-store.md). Applied by
-- `PostgresBackend::migrate`, which embeds this file, so there is one copy of the
-- DDL and `psql -f` gives the same result.
--
-- Four tables, one per space the store owns. The log's primary key is the
-- compare-and-set: a second writer at the same head loses on a unique violation,
-- which is what makes a lease a lease and keeps split brain from forking a log.

create table if not exists store_object (
    zone  text  not null,
    hash  text  not null,
    bytes bytea not null,
    primary key (zone, hash)
);

create table if not exists store_log (
    zone  text   not null,
    log   text   not null,
    seq   bigint not null,
    bytes bytea  not null,
    primary key (zone, log, seq)
);

create table if not exists store_index (
    zone  text  not null,
    key   text  not null,
    value bytea not null,
    primary key (zone, key)
);

-- Prefix scans must use the index whatever the database collation is, hence
-- text_pattern_ops.
create index if not exists store_index_prefix
    on store_index (zone, key text_pattern_ops);

-- The genealogy adjacency, maintained beside the index projection so that lineage
-- and subtree traversals are one recursive CTE instead of a round trip per hop.
-- Rebuildable from the `<shard>+graph` logs like every other projection.
create table if not exists store_edge (
    zone       text not null,
    child      text not null,
    idx        int  not null,
    parent     text not null,
    provenance text not null,
    shard      text not null,
    primary key (zone, child, idx)
);

create index if not exists store_edge_parent on store_edge (zone, parent);
