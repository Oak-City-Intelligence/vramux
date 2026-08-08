# Security

vramux is a single-operator tool that arbitrates one machine's GPU. Its threat
model assumes everything that can reach it is already trusted, and it is worth
saying exactly what that buys you before you run it.

## Supported versions

Pre-1.0. Only the tip of `main` is supported; there are no maintenance
branches and no backports.

## Reporting a vulnerability

Open a private security advisory on the GitHub repository
(**Security → Report a vulnerability**), or mail
`john@oakcityintelligence.com`. Please do not open a public issue for anything
that lets somebody reach memory, processes or files they should not.

Expect an acknowledgement within a week. This is one person's project, so
that is a good-faith target and not a service level.

## What is deliberately not defended

These are design positions, not bugs. Reporting them is welcome as a
discussion; they will not be treated as vulnerabilities.

- **There is no authentication.** Anything that can reach the port can list
  models, run inference, load a model, take a lease, and evict what somebody
  else is using. vramux binds `127.0.0.1` by default and that default is the
  security boundary. Binding it to a routable address publishes an unauthenticated
  remote code path onto your GPU.

- **`models.yml` is a trust boundary.** A `docker` entry names a compose file
  that vramux runs; a `llama-server` entry names a binary path and
  `extra_args` that vramux passes through verbatim. Whoever can write the
  registry chooses what processes start as you. Treat it with the same care as
  a shell profile.

- **The docker backend needs membership in the `docker` group**, which is
  root-equivalent on most systems: that group can bind-mount any path into a
  privileged container. If that is not acceptable on your machine, run the
  `llama-server` backend only and register no `docker` models.

- **Lease denial-of-service is trivial and unmitigated.** One client can take
  a lease covering most of the card and renew it forever, and nothing arbitrates
  between holders. A `priority` field is accepted and currently means nothing.
  This is acceptable on a box with one operator and is a direct reason vramux
  is not a multi-tenant tool.

- **A lease is a promise, not a fence.** vramux reserves memory in its own
  accounting; it does not stop an unmigrated consumer from allocating it
  anyway. Nothing on a GPU can, short of the driver.

- **Backends are not sandboxed.** They run as you, with your environment, and
  vramux's job is starting and stopping them rather than confining them.

- **`/gpu/console` is a page on an unauthenticated port.** It is read-only —
  it renders `/gpu/events` and calls nothing that changes state — and it is
  served from a file in the checkout with no assets fetched from anywhere. It
  adds no capability the API did not already grant to whoever can reach the
  port, which is the same loopback boundary as everything else here. No CORS
  headers are set, so a page on another origin cannot read the stream.

## What is defended, and what to report

- **Command construction.** Backends are spawned as argument vectors, never
  through a shell. A model tag, a served name or a file path that reaches a
  shell — or escapes its argument — is a bug worth reporting.

- **Path handling.** Registry paths are expanded and used as given; a request
  field that reaches the filesystem, or a model tag that traverses out of a
  configured directory, is a bug.

- **Process attribution and reclaim.** vramux resolves who owns GPU memory by
  walking the process tree. A way to make it attribute somebody else's memory to
  your lease — and so be granted memory that is not free — is a bug: it turns
  into an out-of-memory kill of an innocent process.

- **Denial of the router itself.** A request that wedges the event loop, or
  that makes the arbiter hold its lock forever, affects every consumer on the
  machine. Report it.

- **The cache directory.** `~/.cache/vramux/` holds measured costs and usage
  history. Anything that writes outside it, or that follows a symlink out of
  it, is a bug.

## Hardening notes

- Leave the bind address on loopback. If a remote client genuinely needs the
  endpoint, terminate it behind something that authenticates — an SSH tunnel or
  a reverse proxy — and keep vramux itself on `127.0.0.1`.
- Keep `models.yml` writable only by the user that runs the router.
- The compose files you register must set no `restart:` policy. That is a
  correctness requirement rather than a security one, but a container that
  re-pins the card after a reboot looks exactly like a compromise at 2 a.m.
- Run it as a user service, not a system one. vramux needs your GPU, your
  models and your docker group; it needs nothing that root would add.
