//! The privacy policy and terms of service (`/privacy`, `/terms`): plain pages anyone can
//! open, signed in or not, as Google asks of an OAuth app (docs/AUTH.md §3). They are
//! written from this Studio's own settings, so they name the cloud providers actually
//! switched on and the session length actually used.

use crate::scheduler::AppState;
use axum::extract::State;
use axum::response::Html;
use std::env;
use std::sync::Arc;

/// Change it whenever the wording below changes.
const LAST_UPDATED: &str = "25 October 2026";

/// The model agents load when they declare none (`backend/aries_agent/config.py`).
const DEFAULT_MODEL: &str = "Qwen/Qwen-Image-2.1";

/// Models whose license allows no commercial use (all three are under the Qwen Research
/// License), named as such in the terms when in use.
const NON_COMMERCIAL: &[&str] = &[
    "Qwen/Qwen-Image-2.1",
    "Qwen/Qwen-Image-2.1-PE-T2I",
    "Qwen/Qwen-Image-2.1-PE-I2I",
];

/// Who runs this Studio and how to reach them; config only, never in the repo.
#[derive(Debug, Clone, Default)]
pub struct LegalConfig {
    /// `ARIES_LEGAL_OPERATOR`: a person's or group's name.
    pub operator: Option<String>,
    /// `ARIES_LEGAL_CONTACT`: an email address or a URL.
    pub contact: Option<String>,
}

impl LegalConfig {
    pub fn from_env() -> Self {
        let get = |k: &str| {
            env::var(k)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        Self {
            operator: get("ARIES_LEGAL_OPERATOR"),
            contact: get("ARIES_LEGAL_CONTACT"),
        }
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// What both pages say about this Studio, escaped for HTML.
struct Facts {
    operator: String,
    /// `operator` to start a sentence with.
    operator_start: String,
    contact: String,
    session_days: Option<u32>,
    providers: Vec<&'static str>,
    /// Model ids the agents declare (Hugging Face repos), or the default before any has.
    models: Vec<String>,
    /// `false`: no agent has registered yet, so `models` is the default.
    models_declared: bool,
}

impl Facts {
    async fn new(state: &AppState) -> Self {
        let legal = &state.config.legal;
        let (operator, operator_start) = match legal.operator.as_deref() {
            Some(op) => (esc(op), esc(op)),
            None => (
                "the person who runs this Studio".to_string(),
                "The person who runs this Studio".to_string(),
            ),
        };
        let contact = match legal.contact.as_deref() {
            Some(c) if c.contains('@') && !c.contains("://") => {
                format!(r#"<a href="mailto:{0}">{0}</a>"#, esc(c))
            }
            Some(c) if c.starts_with("https://") || c.starts_with("http://") => {
                format!(r#"<a href="{0}">{0}</a>"#, esc(c))
            }
            Some(c) => esc(c),
            None => "the person who invited you".into(),
        };
        let mut providers = Vec::new();
        for info in state.cloud.infos() {
            let name = match info.provider.as_str() {
                "anthropic" => "Anthropic (Claude)",
                "google" => "Google (Gemini)",
                _ => continue,
            };
            if !providers.contains(&name) {
                providers.push(name);
            }
        }
        let mut models: Vec<String> = Vec::new();
        for agent in state.db.list_agents().await.unwrap_or_default() {
            for m in agent.models {
                if !models.contains(&m.id) {
                    models.push(m.id);
                }
            }
        }
        let models_declared = !models.is_empty();
        if !models_declared {
            models.push(DEFAULT_MODEL.to_string());
        }
        Self {
            operator,
            operator_start,
            contact,
            session_days: state.config.auth.as_ref().map(|a| a.session_days),
            providers,
            models,
            models_declared,
        }
    }

    /// The models as links to their model cards, where their licenses are.
    fn model_links(&self) -> String {
        self.models
            .iter()
            .map(|m| format!(r#"<a href="https://huggingface.co/{0}">{0}</a>"#, esc(m)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// What is known about the models' licenses, as a sentence; empty when nothing is.
    fn model_notes(&self) -> String {
        let ids: Vec<String> = self
            .models
            .iter()
            .filter(|m| NON_COMMERCIAL.contains(&m.as_str()))
            .map(|m| esc(m))
            .collect();
        match ids.as_slice() {
            [] => String::new(),
            [one] => format!(" In particular, {one} does not allow commercial use."),
            [rest @ .., last] => format!(
                " In particular, {} and {last} do not allow commercial use.",
                rest.join(", ")
            ),
        }
    }

    fn cloud_sentence(&self) -> String {
        if self.providers.is_empty() {
            "Prompt rewriting runs on hardware the host controls.".into()
        } else {
            format!(
                "If you pick a cloud prompt enhancer ({}), your prompt and any reference images \
                 you attached are sent to that provider to be rewritten, under the host's \
                 account with it, and its terms and privacy policy apply to them. Only the \
                 prompt and the images are sent: not your email address, name or any account \
                 identifier. The built-in enhancer runs on the host's own hardware.",
                self.providers.join(", ")
            )
        }
    }
}

fn page(title: &str, body: &str) -> Html<String> {
    Html(format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Aries Image Studio</title>
<style>
:root {{ --bg:#fcf9f2; --fg:#1c1c18; --muted:#58423a; --line:#e5e2db; --accent:#a53705; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141412; --fg:#f3f0e9; --muted:#a09088; --line:#2d2d2a; --accent:#e0703c; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width:44rem; margin:0 auto; padding:2.5rem 1rem 4rem; }}
h1 {{ font-size:1.75rem; line-height:1.25; margin:0 0 .25rem; }}
h2 {{ font-size:1.1rem; margin:2rem 0 .5rem; }}
p, li {{ color:var(--fg); }}
ul {{ padding-left:1.25rem; }}
li {{ margin:.35rem 0; }}
a {{ color:var(--accent); }}
.updated {{ color:var(--muted); font-size:.9rem; margin:0 0 1.5rem; }}
.notice {{ border-left:3px solid var(--accent); padding:.75rem 1rem; margin:0 0 1.5rem;
  background:color-mix(in srgb, var(--accent) 8%, transparent); border-radius:0 .5rem .5rem 0; }}
nav {{ display:flex; gap:1rem; font-size:.9rem; padding-bottom:1rem; margin-bottom:1.5rem;
  border-bottom:1px solid var(--line); }}
</style>
</head>
<body>
<main>
<nav><a href="/">Aries Image Studio</a><a href="/privacy">Privacy</a><a href="/terms">Terms</a></nav>
<h1>{title}</h1>
<p class="updated">Last updated {LAST_UPDATED}</p>
{body}
</main>
</body>
</html>
"#
    ))
}

pub async fn privacy(State(state): State<Arc<AppState>>) -> Html<String> {
    let f = Facts::new(&state).await;
    let session = match f.session_days {
        Some(d) => format!(
            "<code>aries_session</code>, a random token that keeps you signed in until you \
             sign out, or until {d} days pass without you using Studio (Studio stores a hash \
             of it). No others: a sign-in in progress is tracked on the server"
        ),
        None => "whatever cookies the current version of Studio uses".into(),
    };
    let body = format!(
        r#"<p class="notice"><strong>This is a self-hosted service, run privately by {op}.</strong>
The host is in full control of the code running on it and of the data it stores. It is not a
commercial service and makes no commitment to privacy. Assume that the host can
see, keep and use anything you put into it, and use it only if you trust the host with all
of that.</p>

<h2>Who is responsible</h2>
<p>{op_start} is the host: they run this Studio, decide who may use it, and make the one commitment
on this page, about your Google account information, by running it with Google sign-in.
Studio is open-source software (Aries Image Studio, MIT license). Its authors are not the
host, have no access to this Studio or to what it stores, and are not responsible for how the
host runs it or uses your data. Contact: {contact}.</p>

<h2>What Studio stores</h2>
<ul>
<li><strong>What you put in and make:</strong> your prompts and settings, rewritten prompts and
the enhancer's reasoning, the images you upload as references, the images Studio generates,
your favorites and your preferences. Each generated PNG carries its prompt and settings inside
the file: if you share an image file, you also share its prompt, the rewritten prompt, its
settings and the models that made it.</li>
<li><strong>Your Google account basics:</strong> your email address, whether Google has verified
it, your name and the address of your profile picture, received when you sign in with Google
(the <code>openid</code>, <code>email</code> and <code>profile</code> scopes).</li>
<li><strong>Cookies:</strong> {session}.</li>
<li><strong>Logs</strong> of sign-ins (with the email address) and of what Studio does, and
anything else the host chooses to record.</li>
</ul>
<p>All of it is kept for as long as the host chooses, possibly indefinitely. You can ask the
host to delete your account and content; the host may or may not do so.</p>

<h2>No commitments about your content</h2>
<p>Apart from your Google account information (below), the host makes no commitment about
anything you put into Studio, anything Studio makes from it, or how you use Studio. In
particular, there is no commitment to:</p>
<ul>
<li>keep your prompts, uploads or images private or confidential;</li>
<li>not track, log, monitor or analyse how you use Studio;</li>
<li>not use your content to train, tune or evaluate AI models;</li>
<li>not share, publish or sell your content;</li>
<li>delete your content when you delete it in Studio, or ever;</li>
<li>tell you about any of this, now or later.</li>
</ul>
<p>The host may look at, keep, copy, change, delete, publish, share and use your content for any
purpose, at the host's sole discretion, and may report content or activity to the authorities
and hand over whatever is stored.</p>

<h2>Your Google account information</h2>
<p>This is the one commitment: your email address, name and picture from Google are used only to
sign you in, to tell your account apart from others and to show who is signed in. They are not
sold, not used for advertising and not passed on, except to the authorities or where the law
requires. This commitment covers your Google account information wherever Studio stores it,
logs included.</p>
<p>Studio's use and transfer of information received from Google APIs adheres to the
<a href="https://developers.google.com/terms/api-services-user-data-policy">Google API Services
User Data Policy</a>, including the Limited Use requirements.</p>

<h2>Other people and services</h2>
<p>As currently set up, Studio shows each account its own images, and everyone signed in sees
how busy the shared GPUs are. The host can change that at any time. {cloud}</p>

<h2>Who may use it</h2>
<p>The host decides who is invited, and is responsible for running this service in compliance
with the laws and regulations that apply to it, including any minimum age for its users.</p>

<h2>Your rights</h2>
<p>Nothing here limits rights you have under the law that applies to you.</p>

<h2>Changes</h2>
<p>This policy may change at any time; this page changes with it and the date at the top
moves.</p>
"#,
        op = f.operator,
        op_start = f.operator_start,
        contact = f.contact,
        cloud = f.cloud_sentence(),
    );
    page("Privacy policy", &body)
}

pub async fn terms(State(state): State<Arc<AppState>>) -> Html<String> {
    let f = Facts::new(&state).await;
    let cloud = if f.providers.is_empty() {
        String::new()
    } else {
        format!(
            " Prompts and images you send to a cloud prompt enhancer ({}) go out under the \
             host's account with that provider, so they must also comply with its usage \
             policies.",
            f.providers.join(", ")
        )
    };
    let body = format!(
        r#"<p class="notice"><strong>This is a self-hosted service, run privately by {op}.</strong>
The host is in full control of the code running on it and of the data it stores. It makes no
commitment to privacy, and you use it entirely at your own risk. Use it only if you trust the
host with everything you put into it.</p>

<p>These terms cover your use of this Aries Image Studio. By signing in you agree to them and
to the <a href="/privacy">privacy policy</a>, which says what the host may do with what you
put in. Contact: {contact}.</p>

<h2>Who is responsible</h2>
<p>{op_start} is the host and the other party to these terms. Studio is open-source software (Aries
Image Studio, MIT license); its authors are not party to these terms and are not responsible
for this Studio, its content, or the host's actions.</p>

<h2>Access</h2>
<p>Studio is invitation only and free. Access is at the host's discretion and can be
changed or withdrawn at any time. The host is responsible for running this service in
compliance with the laws and regulations that apply to it, including any minimum age for its
users. Keep your access to yourself: don't share your account or
let others use your session.</p>

<h2>Acceptable use</h2>
<p>Use Studio lawfully and decently. In particular, do not use it to make, upload or request:</p>
<ul>
<li>anything <a href="/terms#zero-tolerance">below</a> (non-consensual intimate imagery, sexual
content involving minors);</li>
<li>content that harasses, threatens or defames someone, or that impersonates a real person to
deceive;</li>
<li>realistic images of a real person, or of real events, that you present or pass on as
real;</li>
<li>material you have no right to use, such as other people's photos or artwork you upload as
references without permission;</li>
<li>anything else that is unlawful where you or the host are.</li>
</ul>
<p>Don't try to get around sign-in, reach other people's images, or overload or break the
service. Breaking these rules can cost you your access, without notice.</p>

<h2 id="zero-tolerance">Intimate images and minors: zero tolerance</h2>
<p>Never use Studio to make, upload or ask for:</p>
<ul>
<li><strong>non-consensual intimate imagery</strong>: nude, sexual or otherwise intimate images of
a real, identifiable person who has not consented to them, whether real, edited or
AI-generated, including "undressing" a photo or putting someone's face on another body;</li>
<li><strong>sexual content involving minors</strong>, real or AI-generated, of any kind, or
anything that sexualises a child.</li>
</ul>
<p>A prompt or upload that asks for these breaks this rule even if no image comes out. These
rules follow the law and Google's own:
the <a href="https://developers.google.com/terms">Google APIs Terms of Service</a>, under which
Studio uses Google sign-in, forbid using Google's services for anything unlawful or that
infringes others' rights, and Google's
<a href="https://policies.google.com/terms/generative-ai/use-policy">Generative AI Prohibited
Use Policy</a> forbids non-consensual intimate imagery and child sexual abuse material.
Studio uses Google sign-in only to tell who is invited, and no Google service makes or edits
its images.</p>
<p><strong>If such content or such a request is discovered, the host may remove your access at
once and report you to the authorities</strong>, with your account details, prompts, uploads
and images, as the law requires or allows. The host has no duty to look for it.</p>

<h2>Your content</h2>
<p>You keep whatever rights you have in your prompts and uploads. By putting them into Studio,
you allow the host to store, look at, copy, share and use them, and what Studio makes from
them, for any purpose, as the <a href="/privacy">privacy policy</a> describes. The host may remove anything, at any
time, without notice. Images made by AI models may not be protected by copyright and can resemble
existing works; how you use them is up to you, and so is the responsibility for it.</p>

<h2>Reporting</h2>
<p>The host may report content or activity that appears unlawful to the authorities, and
cooperate with them, including by handing over your account details, prompts and images.</p>

<h2>Other services</h2>
<p>Signing in uses your Google account under Google's terms.{cloud}</p>

<h2>The models and their terms</h2>
<p>Studio makes images, and rewrites prompts, with AI models the host chooses: {models_when}
{models}. Each model comes with its own license and terms, which apply to what you make with
it, on top of these.{notes} You agree to use what Studio makes only as the license of the model
that made it allows; read a model's terms before you use what it makes, above all for
anything commercial.</p>
<p>The host may change the models at any time, and the terms that apply change with them.
Images made earlier stay under the terms of the model that made them: each image names the
models that made it, with their versions, under Provenance in its details and inside the PNG
file.</p>

<h2>No guarantees</h2>
<p>Studio is provided as is, without warranties of any kind. It may be slow, unavailable, or
lose data, so keep copies of images that matter to you. To the extent the law allows, the
host is not liable for any loss or damage arising from your use of Studio.</p>

<h2>Your rights</h2>
<p>Nothing here limits rights you have under the law that applies to you.</p>

<h2>Changes</h2>
<p>These terms may change; this page changes with them and the date at the top moves. Using
Studio after a change means you accept the new terms.</p>
"#,
        op = f.operator,
        op_start = f.operator_start,
        contact = f.contact,
        models_when = if f.models_declared {
            "currently"
        } else {
            "by default"
        },
        models = f.model_links(),
        notes = f.model_notes(),
    );
    page("Terms of service", &body)
}
