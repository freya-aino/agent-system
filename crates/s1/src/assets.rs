//! Serving the management UI's static assets.
//!
//! The SPA is built in the devenv (`web/`) and served from disk by this binary.
//! Two rules, and both are about not serving what was not meant to be served:
//!
//! **A path that escapes the asset root is refused rather than resolved.** The
//! check is on the *requested* path, before any filesystem call, because
//! canonicalising first would follow a symlink somebody planted — the same reason
//! the Firecracker driver refuses rather than canonicalises.
//!
//! **An unknown path inside the SPA falls back to `index.html`**, because a
//! single-page app owns its own routing, but a request for a *file* that does not
//! exist (anything with an extension) is a 404: silently answering an HTML page to
//! a request for `app.js` turns a build mistake into a mystery.

use std::path::{Component, Path, PathBuf};

use crate::error::{Result, S1Error};

/// What to serve for one asset request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub path: PathBuf,
    pub content_type: &'static str,
}

/// The content type for an extension. A short table: anything a built SPA emits,
/// and nothing else — an unknown extension is served as an opaque download rather
/// than as something a browser will execute.
pub fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("woff2") => "font/woff2",
        Some("map") => "application/json",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

/// Whether a requested path stays inside the root.
pub fn is_safe(requested: &str) -> bool {
    if requested.starts_with('/') || requested.contains('\0') {
        return false;
    }
    Path::new(requested)
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
}

/// Resolve one request against the asset root.
///
/// `None` for the root means this deployment is headless, which is a valid
/// configuration: the API is the product, and the UI is optional.
pub fn resolve(root: Option<&Path>, requested: &str) -> Result<Asset> {
    let root =
        root.ok_or_else(|| S1Error::not_found("asset", "this deployment serves no web interface"))?;
    if !is_safe(requested) {
        // Refused, not normalised: a path that tries to leave the root is not a
        // typo to be corrected.
        return Err(S1Error::bad_request(
            "asset",
            format!("{requested:?} does not stay inside the asset root"),
        ));
    }
    let candidate = root.join(requested);
    if candidate.is_file() {
        return Ok(Asset {
            content_type: content_type(&candidate),
            path: candidate,
        });
    }
    // A request for a file that is not there is a 404; a request for a route is
    // the SPA's own.
    if Path::new(requested).extension().is_some() {
        return Err(S1Error::not_found("asset", requested));
    }
    let index = root.join("index.html");
    if index.is_file() {
        Ok(Asset {
            content_type: "text/html; charset=utf-8",
            path: index,
        })
    } else {
        Err(S1Error::not_found("asset", "index.html"))
    }
}

/// The page a deployment serves when the UI has not been built.
///
/// Honest rather than blank: it says what to run, because a developer who sees an
/// empty page assumes the service is broken.
pub const UNBUILT_PAGE: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>agent-system S1</title>
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font:16px/1.5 system-ui;margin:3rem auto;max-width:40rem">
<h1>S1 is running; the web interface is not built</h1>
<p>The API is up. To build the management UI:</p>
<pre>cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>Then start S1 with <code>web_root</code> pointing at <code>web/dist</code>,
or set <code>AGENT_S1_WEB_ROOT</code>.</p>
<p><a href="/openapi.json">The API document</a> is served either way.</p>
</body></html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("agent-s1-assets-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("assets")).unwrap();
        std::fs::write(dir.join("index.html"), b"<!doctype html>").unwrap();
        std::fs::write(dir.join("assets/app.js"), b"export {}").unwrap();
        dir
    }

    #[test]
    fn a_built_asset_is_served_with_its_type() {
        let root = root();
        let asset = resolve(Some(&root), "assets/app.js").unwrap();
        assert_eq!(asset.path, root.join("assets/app.js"));
        assert_eq!(asset.content_type, "text/javascript; charset=utf-8");
        assert_eq!(
            resolve(Some(&root), "index.html").unwrap().content_type,
            "text/html; charset=utf-8"
        );
    }

    #[test]
    fn a_path_that_tries_to_leave_the_root_is_refused_rather_than_resolved() {
        let root = root();
        for requested in [
            "../../etc/passwd",
            "assets/../../etc/passwd",
            "/etc/passwd",
            "./../secret",
        ] {
            assert!(
                resolve(Some(&root), requested).is_err(),
                "{requested}: canonicalising first would follow a planted symlink"
            );
            assert!(!is_safe(requested));
        }
        assert!(is_safe("assets/app.js"));
        assert!(is_safe("index.html"));
    }

    #[test]
    fn an_spa_route_falls_back_to_the_page_and_a_missing_file_does_not() {
        let root = root();
        let asset = resolve(Some(&root), "deployments/sha256:aa").unwrap();
        assert_eq!(asset.path, root.join("index.html"), "the SPA owns routing");
        assert!(
            resolve(Some(&root), "assets/missing.js").is_err(),
            "answering HTML to a request for a script turns a build mistake into a \
             mystery"
        );
    }

    #[test]
    fn a_headless_deployment_serves_no_assets_and_says_so() {
        let err = resolve(None, "index.html").unwrap_err();
        assert_eq!(err.status(), 404);
        assert!(format!("{err}").contains("no web interface"));
    }

    #[test]
    fn an_unknown_extension_is_not_something_a_browser_will_execute() {
        assert_eq!(
            content_type(Path::new("x.wasm")),
            "application/octet-stream"
        );
        assert_eq!(content_type(Path::new("noext")), "application/octet-stream");
    }

    #[test]
    fn the_unbuilt_page_says_what_to_run() {
        assert!(UNBUILT_PAGE.contains("npm run build"));
        assert!(UNBUILT_PAGE.contains("/openapi.json"));
    }
}
