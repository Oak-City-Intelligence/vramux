#!/usr/bin/env python3
"""Allocate real VRAM and hold it, so the broker has something to arbitrate.

Multi-residency cannot be tested with the models on a workstation that was
tuned for quality: they are all large, so any two of them fill the card and the
interesting case — several modest consumers coexisting — never happens. This
manufactures that case. It is a consumer, not a fake: the memory is a real CUDA
allocation, NVML reports it against this process, and vramux attributes it the
same way it attributes anything else.

    hold_vram.py 2048                    # take 2 GB, hold until Ctrl-C
    hold_vram.py 2048 --seconds 120      # take 2 GB, exit after two minutes
    hold_vram.py 2048 --lease sd-a       # take it under a lease of that owner

Deliberately stdlib-only, through `libcuda` with ctypes. Pulling in torch to
allocate a buffer would make the harness heavier than the thing it tests, and
would not run on a machine that has no torch.
"""
import argparse
import ctypes
import os
import signal
import sys
import time
import urllib.request
import json

MB = 1024 * 1024

# The handful of driver calls this needs. The CUDA driver API is stable and
# these four have not changed signature in a decade.
CUDA_SUCCESS = 0


def _load_driver():
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    sys.exit("libcuda not found — this needs an NVIDIA driver, not the CUDA toolkit")


def _check(cuda, code, what):
    if code != CUDA_SUCCESS:
        msg = ctypes.c_char_p()
        try:
            cuda.cuGetErrorString(code, ctypes.byref(msg))
            detail = msg.value.decode() if msg.value else str(code)
        except Exception:
            detail = str(code)
        sys.exit("%s failed: %s" % (what, detail))


class Lease:
    """A lease held for as long as this object is alive, renewed on a timer.

    The renew loop is the point: a TTL long enough to outlive the hold would be
    the wrong shape to test, because a real holder that dies must lose its
    lease. Renewal is what proves the holder is still there.
    """

    def __init__(self, base, owner, mb, ttl=60):
        self.base = base.rstrip("/")
        self.owner = owner
        self.mb = mb
        self.ttl = ttl
        self.id = None
        self.next_renew = 0.0

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)

    def acquire(self):
        # Send our own pid: this process is about to allocate the memory it is
        # asking for, and without the pid the broker charges the card twice —
        # once for the grant, once for the allocation it can see.
        body = self._post("/gpu/lease", {
            "mb": self.mb,
            "owner": self.owner,
            "ttl": self.ttl,
            "pid": os.getpid(),
        })
        # The grant reports its id as `lease`, not `id`. Reading the wrong key
        # is quiet: acquire still succeeds, and only release turns into a no-op
        # that leaves the grant to time out.
        self.id = body.get("lease") or body.get("id")
        self.next_renew = time.monotonic() + self.ttl / 3
        print("lease %s granted to %s for %d MB" % (self.id, self.owner, self.mb))
        return self

    def tick(self):
        if not self.id or time.monotonic() < self.next_renew:
            return
        try:
            self._post("/gpu/lease/%s/renew" % self.id, {})
            self.next_renew = time.monotonic() + self.ttl / 3
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Expired or dropped by a restart. Re-acquiring is safe and
                # charges nothing new, because the memory we hold is already
                # attributed to this pid.
                print("lease gone (404) — re-acquiring")
                self.acquire()
            else:
                raise

    def release(self):
        if not self.id:
            return
        req = urllib.request.Request(
            "%s/gpu/lease/%s" % (self.base, self.id), method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=10).close()
            print("lease %s released" % self.id)
        except Exception as exc:
            print("could not release %s: %s" % (self.id, exc))
        self.id = None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mb", type=int, help="how much VRAM to hold, in MiB")
    ap.add_argument("--seconds", type=float, default=None,
                    help="release and exit after this long (default: until killed)")
    ap.add_argument("--lease", metavar="OWNER", default=None,
                    help="hold it under a lease with this owner name")
    ap.add_argument("--ttl", type=int, default=60, help="lease TTL in seconds")
    ap.add_argument("--url", default=os.environ.get("VRAMUX_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--device", type=int,
                    default=int(os.environ.get("VRAMUX_DEVICE", "0")))
    args = ap.parse_args()

    lease = None
    if args.lease:
        # Ask before allocating, which is the order a well-behaved consumer
        # uses: a grant that arrives after the allocation is just bookkeeping.
        lease = Lease(args.url, args.lease, args.mb, args.ttl).acquire()

    cuda = _load_driver()
    _check(cuda, cuda.cuInit(0), "cuInit")

    dev = ctypes.c_int()
    _check(cuda, cuda.cuDeviceGet(ctypes.byref(dev), args.device), "cuDeviceGet")

    ctx = ctypes.c_void_p()
    _check(cuda, cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    # The context itself costs a few hundred MB before a single byte is asked
    # for. That is exactly what VRAMUX_RESERVE_MB exists to cover, and holding
    # it here is part of what makes this a realistic consumer.
    ptr = ctypes.c_void_p()
    _check(cuda, cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(args.mb * MB)),
           "cuMemAlloc(%d MB)" % args.mb)

    print("holding %d MB on device %d (pid %d)" % (args.mb, args.device, os.getpid()))

    stop = {"now": False}

    def _stop(signum, frame):
        stop["now"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while not stop["now"]:
            if deadline and time.monotonic() >= deadline:
                break
            if lease:
                lease.tick()
            time.sleep(0.5)
    finally:
        # Release the lease before freeing the memory, never after: the other
        # order leaves a window where the broker believes memory is reserved
        # that the card has already handed back.
        if lease:
            lease.release()
        cuda.cuMemFree_v2(ptr)
        cuda.cuCtxDestroy_v2(ctx)
        print("released %d MB" % args.mb)


if __name__ == "__main__":
    main()
