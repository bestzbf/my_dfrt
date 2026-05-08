#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  /data_cos/cos_systemd_download.sh [options] <source> <dest_dir>

Source can be either:
  /data_cos/path/to/dir
  cos://bucket-name/path/to/dir

By default, the script downloads <source> into <dest_dir>/<basename(source)>/.
Use --exact-dst if <dest_dir> already includes the final directory name.

Options:
  --unit NAME             systemd unit name. Default: generated unique name.
  --bucket NAME           bucket used when source starts with /data_cos.
                          Default: hd-ai-data-1251882982
  --routines N            file-level concurrency. Default: 64
  --thread-num N          multipart concurrency per file. Default: 4
  --log-root DIR          log directory. Default: /data2/d4rt/cos_download_logs
  --snapshot-root DIR     snapshot directory. Default: /data2/d4rt/.coscli_snapshot
  --exact-dst             do not append basename(source) to dest_dir.
  --no-update             do not pass --update to coscli sync.
  --print-only            print resolved paths and exit without starting download.
  -h, --help              show this help.

Examples:
  # Download /data_cos/hdu_datasets/scannet/scans_test to /data3/dataset/scannet/scans_test
  /data_cos/cos_systemd_download.sh \
    /data_cos/hdu_datasets/scannet/scans_test \
    /data3/dataset/scannet

  # Download with exact destination (no basename appending)
  /data_cos/cos_systemd_download.sh --exact-dst \
    cos://hd-ai-data-1251882982/hdu_datasets/foo \
    /data3/dataset/foo
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

shell_safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '-' | sed 's/^-//; s/-$//'
}

ensure_trailing_slash() {
  local value="$1"
  value="${value%/}"
  printf '%s/' "$value"
}

bucket="hd-ai-data-1251882982"
routines="64"
thread_num="4"
log_root="/data2/d4rt/cos_download_logs"
snapshot_root="/data2/d4rt/.coscli_snapshot"
unit=""
append_basename=1
use_update=1
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit)
      [[ $# -ge 2 ]] || die "--unit requires a value"
      unit="$2"
      shift 2
      ;;
    --bucket)
      [[ $# -ge 2 ]] || die "--bucket requires a value"
      bucket="$2"
      shift 2
      ;;
    --routines)
      [[ $# -ge 2 ]] || die "--routines requires a value"
      routines="$2"
      shift 2
      ;;
    --thread-num)
      [[ $# -ge 2 ]] || die "--thread-num requires a value"
      thread_num="$2"
      shift 2
      ;;
    --log-root)
      [[ $# -ge 2 ]] || die "--log-root requires a value"
      log_root="$2"
      shift 2
      ;;
    --snapshot-root)
      [[ $# -ge 2 ]] || die "--snapshot-root requires a value"
      snapshot_root="$2"
      shift 2
      ;;
    --exact-dst)
      append_basename=0
      shift
      ;;
    --no-update)
      use_update=0
      shift
      ;;
    --print-only)
      print_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -eq 2 ]] || { usage >&2; exit 2; }

src="${1%/}"
dst="${2%/}"

command -v coscli >/dev/null 2>&1 || die "coscli not found in PATH"
coscli_bin="$(command -v coscli)"
command -v systemd-run >/dev/null 2>&1 || die "systemd-run not found in PATH"

# Convert source to cos:// URL
if [[ "$src" == cos://* ]]; then
  cos_src="${src%/}"
elif [[ "$src" == /data_cos || "$src" == /data_cos/* ]]; then
  prefix="${src#/data_cos}"
  prefix="${prefix#/}"
  if [[ -n "$prefix" ]]; then
    cos_src="cos://${bucket}/${prefix}"
  else
    cos_src="cos://${bucket}"
  fi
else
  die "source must start with /data_cos or cos://: $src"
fi

src_base="$(basename "$cos_src")"

# Destination must be a local path
if [[ "$dst" == cos://* ]] || [[ "$dst" == /data_cos || "$dst" == /data_cos/* ]]; then
  die "destination must be a local path (not /data_cos or cos://): $dst"
fi

if [[ "$append_basename" -eq 1 ]]; then
  dst_base="$(basename "$dst")"
  if [[ "$dst_base" != "$src_base" ]]; then
    dst="${dst%/}/${src_base}"
  fi
fi

cos_src_slash="$(ensure_trailing_slash "$cos_src")"
dst_slash="$(ensure_trailing_slash "$dst")"

safe_base="$(shell_safe_name "$src_base")"
timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$unit" ]]; then
  unit="cos-download-${safe_base}-${timestamp}"
fi
unit="$(shell_safe_name "$unit")"

hash_key="$(printf '%s\n%s\n' "$cos_src_slash" "$dst_slash" | sha1sum | awk '{print $1}')"
snapshot_path="${snapshot_root}/${safe_base}_${hash_key:0:12}"
process_log_path="${log_root}/process"
fail_output_path="${log_root}/failed"
main_log="${log_root}/download_${safe_base}_${timestamp}.log"

update_arg=""
if [[ "$use_update" -eq 1 ]]; then
  update_arg="--update"
fi

echo "Source:      $cos_src_slash"
echo "Destination: $dst_slash"
echo "Unit:        $unit"
echo "Main log:    $main_log"
echo "Process log: $process_log_path"
echo "Snapshot:    $snapshot_path"
echo "Concurrency: --routines $routines --thread-num $thread_num"
if [[ "$use_update" -eq 1 ]]; then
  echo "Mode:        sync with --update"
else
  echo "Mode:        sync without --update"
fi

if [[ "$print_only" -eq 1 ]]; then
  exit 0
fi

systemd_args=()
systemctl_args=()
if [[ "${COS_SYSTEMD_USER:-0}" == "1" ]]; then
  systemd_args+=(--user)
  systemctl_args+=(--user)
fi

mkdir -p "$log_root" "$process_log_path" "$fail_output_path" "$snapshot_root"
mkdir -p "$(dirname "$dst_slash")"
touch "$main_log"

if systemctl "${systemctl_args[@]}" list-units --full --all "${unit}.service" --no-legend 2>/dev/null | grep -q .; then
  die "systemd unit already exists: ${unit}. Stop it first or pass --unit with another name."
fi

systemd-run \
  "${systemd_args[@]}" \
  --unit="$unit" \
  --description="COS download ${src_base}" \
  --setenv=HOME="$HOME" \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv=COS_DOWNLOAD_SRC="$cos_src_slash" \
  --setenv=COS_DOWNLOAD_DST="$dst_slash" \
  --setenv=COS_DOWNLOAD_LOG="$main_log" \
  --setenv=COS_DOWNLOAD_SNAPSHOT="$snapshot_path" \
  --setenv=COS_DOWNLOAD_PROCESS_LOG="$process_log_path" \
  --setenv=COS_DOWNLOAD_FAIL_LOG="$fail_output_path" \
  --setenv=COS_DOWNLOAD_ROUTINES="$routines" \
  --setenv=COS_DOWNLOAD_THREAD_NUM="$thread_num" \
  --setenv=COS_DOWNLOAD_UPDATE_ARG="$update_arg" \
  --setenv=COSCLI_BIN="$coscli_bin" \
  --collect \
  /bin/bash -c '
    mkdir -p "$COS_DOWNLOAD_PROCESS_LOG" "$COS_DOWNLOAD_FAIL_LOG" "$(dirname "$COS_DOWNLOAD_LOG")" "$(dirname "$COS_DOWNLOAD_SNAPSHOT")" "$COS_DOWNLOAD_DST"
    exec "$COSCLI_BIN" --init-skip sync \
      "$COS_DOWNLOAD_SRC" \
      "$COS_DOWNLOAD_DST" \
      -r \
      --routines "$COS_DOWNLOAD_ROUTINES" \
      --thread-num "$COS_DOWNLOAD_THREAD_NUM" \
      $COS_DOWNLOAD_UPDATE_ARG \
      --snapshot-path "$COS_DOWNLOAD_SNAPSHOT" \
      --process-log-path "$COS_DOWNLOAD_PROCESS_LOG" \
      --fail-output-path "$COS_DOWNLOAD_FAIL_LOG" \
      >> "$COS_DOWNLOAD_LOG" 2>&1
  '

echo
echo "Showing progress. Press Ctrl-C to stop watching only; download keeps running under systemd."
echo "Check later with: systemctl status --no-pager $unit"
echo

tail -n +1 -f "$main_log" | tr '\r' '\n' &
tail_pid=$!

cleanup_tail() {
  kill "$tail_pid" >/dev/null 2>&1 || true
}
trap cleanup_tail EXIT INT TERM

while systemctl "${systemctl_args[@]}" is-active --quiet "$unit"; do
  sleep 2
done

sleep 1
cleanup_tail
wait "$tail_pid" 2>/dev/null || true

echo
systemctl status --no-pager "$unit" 2>/dev/null || true

if grep -q ' Succeed:' "$main_log" || grep -q '^Succeed:' "$main_log"; then
  echo
  echo "Download finished successfully."
  exit 0
fi

if grep -qiE 'ERRO|failed|error' "$main_log"; then
  echo
  echo "Download finished or stopped with errors. Check: $main_log" >&2
  exit 1
fi

echo
echo "Download service is no longer active. Check log: $main_log"
