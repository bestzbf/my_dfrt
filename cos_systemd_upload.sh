#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./cos_systemd_upload.sh [options] <source_dir> <dest>

Dest can be either:
  /data_cos/path/to/parent
  cos://bucket-name/path/to/parent

By default, the script uploads <source_dir> into <dest>/<basename(source_dir)>/.
Use --exact-dst if <dest> already includes the final directory name.

Options:
  --unit NAME             systemd unit name. Default: generated unique name.
  --bucket NAME           bucket used when dest starts with /data_cos.
                          Default: hd-ai-data-1251882982
  --routines N            file-level concurrency. Default: 64
  --thread-num N          multipart concurrency per file. Default: 4
  --log-root DIR          log directory. Default: /data2/d4rt/cos_upload_logs
  --snapshot-root DIR     snapshot directory. Default: /data2/d4rt/.coscli_snapshot
  --exact-dst             do not append basename(source_dir) to dest.
  --no-update             do not pass --update to coscli sync.
  --print-only            print resolved paths and exit without starting upload.
  -h, --help              show this help.

Examples:
  ./cos_systemd_upload.sh \
    /data/re10k/re10k/processed_v2/train_5000_t81_592x336 \
    /data_cos/data_cos/data_cos/hdu_datasets

  ./cos_systemd_upload.sh --exact-dst \
    /data/src/foo \
    cos://hd-ai-data-1251882982/hdu_datasets/foo/
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
log_root="/data2/d4rt/cos_upload_logs"
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

[[ -d "$src" ]] || die "source directory does not exist: $src"
command -v coscli >/dev/null 2>&1 || die "coscli not found in PATH"
command -v systemd-run >/dev/null 2>&1 || die "systemd-run not found in PATH"

src_base="$(basename "$src")"

if [[ "$dst" == cos://* ]]; then
  cos_dst="${dst%/}"
elif [[ "$dst" == /data_cos || "$dst" == /data_cos/* ]]; then
  prefix="${dst#/data_cos}"
  prefix="${prefix#/}"
  if [[ -n "$prefix" ]]; then
    cos_dst="cos://${bucket}/${prefix}"
  else
    cos_dst="cos://${bucket}"
  fi
else
  die "dest must start with /data_cos or cos://: $dst"
fi

if [[ "$append_basename" -eq 1 ]]; then
  dst_base="$(basename "$cos_dst")"
  if [[ "$dst_base" != "$src_base" ]]; then
    cos_dst="${cos_dst%/}/${src_base}"
  fi
fi

src_slash="$(ensure_trailing_slash "$src")"
cos_dst_slash="$(ensure_trailing_slash "$cos_dst")"

safe_base="$(shell_safe_name "$src_base")"
timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$unit" ]]; then
  unit="cos-upload-${safe_base}-${timestamp}"
fi
unit="$(shell_safe_name "$unit")"

hash_key="$(printf '%s\n%s\n' "$src_slash" "$cos_dst_slash" | sha1sum | awk '{print $1}')"
snapshot_path="${snapshot_root}/${safe_base}_${hash_key:0:12}"
process_log_path="${log_root}/process"
fail_output_path="${log_root}/failed"
main_log="${log_root}/upload_${safe_base}_${timestamp}.log"

update_arg=""
if [[ "$use_update" -eq 1 ]]; then
  update_arg="--update"
fi

echo "Source:      $src_slash"
echo "Destination: $cos_dst_slash"
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

mkdir -p "$log_root" "$process_log_path" "$fail_output_path" "$snapshot_root"
touch "$main_log"

if systemctl list-units --full --all "${unit}.service" --no-legend 2>/dev/null | grep -q .; then
  die "systemd unit already exists: ${unit}. Stop it first or pass --unit with another name."
fi

systemd-run \
  --unit="$unit" \
  --description="COS upload ${src_base}" \
  --setenv=HOME=/root \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv=COS_UPLOAD_SRC="$src_slash" \
  --setenv=COS_UPLOAD_DST="$cos_dst_slash" \
  --setenv=COS_UPLOAD_LOG="$main_log" \
  --setenv=COS_UPLOAD_SNAPSHOT="$snapshot_path" \
  --setenv=COS_UPLOAD_PROCESS_LOG="$process_log_path" \
  --setenv=COS_UPLOAD_FAIL_LOG="$fail_output_path" \
  --setenv=COS_UPLOAD_ROUTINES="$routines" \
  --setenv=COS_UPLOAD_THREAD_NUM="$thread_num" \
  --setenv=COS_UPLOAD_UPDATE_ARG="$update_arg" \
  --collect \
  /bin/bash -c '
    mkdir -p "$COS_UPLOAD_PROCESS_LOG" "$COS_UPLOAD_FAIL_LOG" "$(dirname "$COS_UPLOAD_LOG")" "$(dirname "$COS_UPLOAD_SNAPSHOT")"
    exec coscli sync \
      "$COS_UPLOAD_SRC" \
      "$COS_UPLOAD_DST" \
      -r \
      --routines "$COS_UPLOAD_ROUTINES" \
      --thread-num "$COS_UPLOAD_THREAD_NUM" \
      $COS_UPLOAD_UPDATE_ARG \
      --snapshot-path "$COS_UPLOAD_SNAPSHOT" \
      --process-log-path "$COS_UPLOAD_PROCESS_LOG" \
      --fail-output-path "$COS_UPLOAD_FAIL_LOG" \
      >> "$COS_UPLOAD_LOG" 2>&1
  '

echo
echo "Showing progress. Press Ctrl-C to stop watching only; upload keeps running under systemd."
echo "Check later with: systemctl status --no-pager $unit"
echo

tail -n +1 -f "$main_log" | tr '\r' '\n' &
tail_pid=$!

cleanup_tail() {
  kill "$tail_pid" >/dev/null 2>&1 || true
}
trap cleanup_tail EXIT INT TERM

while systemctl is-active --quiet "$unit"; do
  sleep 2
done

sleep 1
cleanup_tail
wait "$tail_pid" 2>/dev/null || true

echo
systemctl status --no-pager "$unit" 2>/dev/null || true

if grep -q ' Succeed:' "$main_log" || grep -q '^Succeed:' "$main_log"; then
  echo
  echo "Upload finished successfully."
  exit 0
fi

if grep -qiE 'ERRO|failed|error' "$main_log"; then
  echo
  echo "Upload finished or stopped with errors. Check: $main_log" >&2
  exit 1
fi

echo
echo "Upload service is no longer active. Check log: $main_log"
