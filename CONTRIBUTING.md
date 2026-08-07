# Contributing

vramux is one operator's setup that grew into a tool. Contributions are
welcome; so is the warning that the maintainer's machine is the only one it has
ever been tuned on, so a change that is obviously right for your card may need
discussion before it lands.

## Before writing code

Read `DESIGN.md` first, and `ROADMAP.md` second. They are not decoration: the
design records arguments that cost real out-of-memory kills to learn, and the
roadmap says which parts are deliberately not built yet.

Two of those arguments come up often enough to state here:

- **The budget is anchored on what the device reports as used**, not on a sum
  of what vramux believes it handed out. `budget.py`'s module docstring is the
  whole argument. A patch that replaces it with a second accounting of declared
  costs will be asked to explain how the two are kept from drifting.

- **One model is resident at a time on purpose.** Admission does not consult
  measured costs yet because the cost dataset is thin, and an underestimate is
  not a slow request — it is an out-of-memory kill that can take the innocent
  resident down with it. Opening that up is a roadmap stage with a measurement
  gate in front of it, not a constant to raise.

If a change touches either, open an issue before the pull request.

## Development setup

```bash
git clone git@github.com:Oak-City-Intelligence/vramux.git
cd vramux
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-dev.txt
cp models.example.yml models.yml   # then edit it
```

`--system-site-packages` is a convenience, not a requirement: it lets the venv
borrow `aiohttp` and `PyYAML` from the system python so only the test
dependencies are installed into it. A fully isolated venv works too, given
those two packages.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite runs in about a second and needs **no GPU, no llama.cpp build, no
docker and no model weights** — every one of those is behind a fake. Keep it
that way. A test that only passes on a machine with a 24 GB card is a test
nobody else can run, and the first thing that stops being run at all.

Bug fixes want a regression test that fails before the fix. Several of the
existing tests exist because a plausible-looking simplification was wrong:
a process named `(my program)`, a `docker compose top` whose columns moved.
If a test looks over-specific, the comment above it usually says which
production incident it is standing in for.

## Pull requests

- One concern per pull request. A behaviour change and a rename in the same
  diff take three times as long to review.
- Say what you observed, not only what you changed. "Free memory dropped by
  the lease amount twice" is reviewable; "improve accounting" is not.
- Note what you ran it against. GPU model, card size, backend kind, and
  whether it ran against a live card at all. "Tests only, no GPU" is a
  perfectly good answer and much better than silence.
- Documentation lives beside the thing it describes. A new environment
  variable belongs in the README table; a new argument belongs in
  `DESIGN.md` only if it changes a decision rather than an implementation.

## Style

Match the file you are editing. Beyond that:

- Standard library first. The router half may use `aiohttp` and `PyYAML`; the
  client half (`cli.py`) is **stdlib-only on purpose**, because it is copied
  onto machines and into containers that have nothing installed. A dependency
  added there breaks that.
- Comments explain why, not what. The codebase is deliberately heavy on the
  reasoning behind a choice and light on narration of the code beneath it.
- Log lines are read at 2 a.m. by somebody trying to work out why their card is
  full. Name the model, the amount and the reason.
- Python 3.10 is the floor.

## Reporting bugs

Include what the card looked like. `python -m vramux state` and the relevant
window of `journalctl --user -u vramux.service` answer most of the questions a
first reply would otherwise have to ask.

Security issues do not go in the issue tracker — see `SECURITY.md`.
