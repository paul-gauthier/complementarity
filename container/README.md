# Reproducible processing container

The container provides the reference environment used to regenerate the
artifact, including versioned Python dependencies, libRadtran, and REPTRAN.

From the repository root:

```sh
./container/build.sh
./container/run.sh
```

The run wrapper mounts the repository at `/workspace`, uses the current user,
places caches under `/tmp`, and invokes `scripts/reproduce.sh`. Supply another
command to run it in the same environment:

```sh
./container/run.sh python -m scripts.verify_results
```

Set `IMAGE` to override the default image tag `complementarity-artifact:local`.
Building the image requires network access; reproducing the artifact does not.
