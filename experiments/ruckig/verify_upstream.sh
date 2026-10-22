#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
count=0
while read -r expected path; do
  actual=$(git hash-object "$here/vendor/$path")
  if [ "$actual" != "$expected" ]; then
    echo "Upstream Git blob mismatch: $path" >&2
    exit 1
  fi
  count=$((count+1))
done < "$here/vendor/git-blobs.txt"
echo "Verified $count official upstream files at a8db97a4e9c55e5160a3855f739fa3b270df8e4c"
