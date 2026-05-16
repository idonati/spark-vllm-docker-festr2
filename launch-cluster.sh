#!/bin/bash

# Default Configuration
IMAGE_NAME="vllm-node"
DEFAULT_CONTAINER_NAME="vllm_node"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
# Modify these if you want to pass additional docker args or set VLLM_SPARK_EXTRA_DOCKER_ARGS variable
DOCKER_ARGS="-e NCCL_IGNORE_CPU_AFFINITY=1 -v $HF_CACHE_DIR:/root/.cache/huggingface"

# Append additional arguments from environment variable
if [[ -n "$VLLM_SPARK_EXTRA_DOCKER_ARGS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS $VLLM_SPARK_EXTRA_DOCKER_ARGS"
fi

# ETH_IF and IB_IF will be auto-detected if not provided
ETH_IF=""
IB_IF=""
NCCL_DEBUG_VAL=""
MASTER_PORT="29501"

# Initialize variables
NODES_ARG=""
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
COMMAND_TO_RUN=""
DAEMON_MODE="false"
CHECK_CONFIG="false"
ACTION="start"
CLUSTER_WAS_RUNNING="false"
MOD_PATHS=()
MOD_TYPES=()
LAUNCH_SCRIPT_PATH=""
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

ACTIONS_ARG=""
SOLO_MODE="false"
NO_RAY_MODE="false"
LAUNCH_SCRIPT_MODE="false"
MOUNT_CACHE_DIRS="true"
BUILD_JOBS=""
NON_PRIVILEGED_MODE="false"
MEM_LIMIT_GB="110"
MEM_SWAP_LIMIT_GB=""
PIDS_LIMIT="4096"
SHM_SIZE_GB="64"

# Function to print usage
usage() {
    echo "Usage: $0 [-n <node_ips>] [-t <image_name>] [--name <container_name>] [--eth-if <if_name>] [--ib-if <if_name>] [--nccl-debug <level>] [--check-config] [--solo] [-d] [action] [command]"
    echo "  -n, --nodes     Comma-separated list of node IPs (Optional, auto-detected if omitted)"
    echo "  -t              Docker image name (Optional, default: $IMAGE_NAME)"
    echo "  --name          Container name (Optional, default: $DEFAULT_CONTAINER_NAME)"
    echo "  --eth-if        Ethernet interface (Optional, auto-detected)"
    echo "  --ib-if         InfiniBand interface (Optional, auto-detected)"
    echo "  -e, --env       Environment variable to pass to container (e.g. -e VAR=val)"
    echo "  -j              Number of parallel jobs for build environment variables (optional)"
    echo "  --nccl-debug    NCCL debug level (Optional, one of: VERSION, WARN, INFO, TRACE). If no level is provided, defaults to INFO."
    echo "  --apply-mod     Path to directory or zip file containing run.sh to apply before launch (Can be specified multiple times)"
    echo "  --launch-script Path to bash script to execute in the container (from examples/ directory or absolute path). If launch script is specified, action should be omitted."
    echo "  --check-config  Check configuration and auto-detection without launching"
    echo "  --solo          Solo mode: skip autodetection, launch only on current node, do not launch Ray cluster"
    echo "  --master-port   Port for cluster coordination: Ray head port or PyTorch distributed master port (default: 29501)"
    echo "  --no-ray        No-Ray mode: run multi-node vLLM without Ray (uses PyTorch distributed backend)"
    echo "  --no-cache-dirs Do not mount default cache directories (~/.cache/vllm, ~/.cache/flashinfer, ~/.triton)"
    echo "  -d              Daemon mode (only for 'start' action)"
    echo "  --non-privileged Run in non-privileged mode (removes --privileged and --ipc=host)"
    echo "  --mem-limit-gb  Memory limit in GB (default: 110, only with --non-privileged)"
    echo "  --mem-swap-limit-gb Memory+swap limit in GB (default: mem-limit + 10, only with --non-privileged)"
    echo "  --pids-limit    Process limit (default: 4096, only with --non-privileged)"
    echo "  --shm-size-gb   Shared memory size in GB (default: 64, only with --non-privileged)"
    echo "  action          start | stop | status | exec (Default: start). Not compatible with --launch-script."
    echo "  command         Command to run (only for 'exec' action). Not compatible with --launch-script."
    echo ""
    echo "Launch Script Usage:"
    echo "  $0 --launch-script examples/my-script.sh   # Script copied to container and executed"
    echo "  $0 --launch-script /path/to/script.sh      # Uses absolute path to script"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--nodes) NODES_ARG="$2"; shift ;;
        -t) IMAGE_NAME="$2"; shift ;;
        --name) CONTAINER_NAME="$2"; shift ;;
        --eth-if) ETH_IF="$2"; shift ;;
        --ib-if) IB_IF="$2"; shift ;;
        -e|--env) DOCKER_ARGS="$DOCKER_ARGS -e $2"; shift ;;
        -j) BUILD_JOBS="$2"; shift ;;
        --apply-mod) MOD_PATHS+=("$2"); shift ;;
        --launch-script) LAUNCH_SCRIPT_PATH="$2"; shift ;;
        --nccl-debug)
            if [[ -n "$2" && "$2" =~ ^(VERSION|WARN|INFO|TRACE)$ ]]; then
                NCCL_DEBUG_VAL="$2"
                shift
            else
                NCCL_DEBUG_VAL="INFO"
            fi
            ;;
        --master-port|--head-port) MASTER_PORT="$2"; shift ;;
        --check-config) CHECK_CONFIG="true" ;;
        --solo) SOLO_MODE="true" ;;
        --no-ray) NO_RAY_MODE="true" ;;
        --no-cache-dirs) MOUNT_CACHE_DIRS="false" ;;
        --non-privileged) NON_PRIVILEGED_MODE="true" ;;
        --mem-limit-gb) MEM_LIMIT_GB="$2"; shift ;;
        --mem-swap-limit-gb) MEM_SWAP_LIMIT_GB="$2"; shift ;;
        --pids-limit) PIDS_LIMIT="$2"; shift ;;
        --shm-size-gb) SHM_SIZE_GB="$2"; shift ;;
        -d) DAEMON_MODE="true" ;;
        -h|--help) usage ;;
        start|stop|status) 
            if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
                echo "Error: Action '$1' is not compatible with --launch-script. Please omit the action or not use --launch-script."
                exit 1
            fi
            ACTION="$1" 
            ;;
        exec)
            if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
                echo "Error: Action 'exec' is not compatible with --launch-script. Please omit the action or not use --launch-script."
                exit 1
            fi
            ACTION="exec"
            shift
            COMMAND_TO_RUN=$(printf "%q " "$@")
            break
            ;;
        *) 
            echo "Error: Unknown argument or action: $1"
            usage
            ;;
    esac
    shift
done

# Validate non-privileged mode flags
if [[ "$NON_PRIVILEGED_MODE" == "true" ]]; then
    # Set default swap limit if not specified
    if [[ -z "$MEM_SWAP_LIMIT_GB" ]]; then
        MEM_SWAP_LIMIT_GB=$((MEM_LIMIT_GB + 10))
    fi
else
    # Check if non-privileged flags were used without --non-privileged
    for flag in "--mem-limit-gb" "--mem-swap-limit-gb" "--pids-limit" "--shm-size-gb"; do
        if [[ "$*" == *"$flag"* ]]; then
            echo "Error: $flag can only be used with --non-privileged"
            exit 1
        fi
    done
fi

# Append NCCL_DEBUG if set, with validation
if [[ -n "$NCCL_DEBUG_VAL" ]]; then
    case "$NCCL_DEBUG_VAL" in
        VERSION|WARN|INFO|TRACE)
            DOCKER_ARGS="$DOCKER_ARGS -e NCCL_DEBUG=$NCCL_DEBUG_VAL"
            ;;
        *)
            echo "Error: Invalid value for --nccl-debug: $NCCL_DEBUG_VAL"
            echo "Allowed values: VERSION, WARN, INFO, TRACE"
            exit 1
            ;;
    esac
fi

# Add build job parallelization environment variables if BUILD_JOBS is set
if [[ -n "$BUILD_JOBS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS -e MAX_JOBS=$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e CMAKE_BUILD_PARALLEL_LEVEL=$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e NINJAFLAGS=-j$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e MAKEFLAGS=-j$BUILD_JOBS"
fi

# Add cache dirs if requested
CACHE_DIRS_TO_CREATE=()
if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
    # vLLM Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.cache/vllm:/root/.cache/vllm"
    CACHE_DIRS_TO_CREATE+=("$HOME/.cache/vllm")
    
    # FlashInfer Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.cache/flashinfer:/root/.cache/flashinfer"
    CACHE_DIRS_TO_CREATE+=("$HOME/.cache/flashinfer")

    # Triton Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.triton:/root/.triton"
    CACHE_DIRS_TO_CREATE+=("$HOME/.triton")
fi

# Resolve launch script path if specified
if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
    # Check if it's an absolute path or relative path that exists
    if [[ -f "$LAUNCH_SCRIPT_PATH" ]]; then
        LAUNCH_SCRIPT_PATH=$(realpath "$LAUNCH_SCRIPT_PATH")
    # Check if it's just a filename, look in examples/ directory
    elif [[ -f "$SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH" ]]; then
        LAUNCH_SCRIPT_PATH="$SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH"
    # Check if it's a name without .sh extension
    elif [[ -f "$SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh" ]]; then
        LAUNCH_SCRIPT_PATH="$SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh"
    else
        echo "Error: Launch script '$LAUNCH_SCRIPT_PATH' not found."
        echo "Searched in:"
        echo "  - $LAUNCH_SCRIPT_PATH"
        echo "  - $SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH"
        echo "  - $SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh"
        exit 1
    fi
    
    echo "Using launch script: $LAUNCH_SCRIPT_PATH"
    
    # Set command to run the copied script (use absolute path since docker exec may not be in /workspace)
    COMMAND_TO_RUN="/workspace/exec-script.sh"
    LAUNCH_SCRIPT_MODE="true"

    # If launch script is specified, default action to exec unless explicitly set to stop/status
    if [[ "$ACTION" == "start" ]]; then
        ACTION="exec"
    fi
fi

# Validate MOD_PATHS if set
for i in "${!MOD_PATHS[@]}"; do
    mod_path="${MOD_PATHS[$i]}"
    if [[ ! -e "$mod_path" ]]; then
        echo "Error: Mod path '$mod_path' does not exist."
        exit 1
    fi
    
    if [[ -d "$mod_path" ]]; then
        if [[ ! -f "$mod_path/run.sh" ]]; then
             echo "Error: Mod directory '$mod_path' must contain 'run.sh'."
             exit 1
        fi
        MOD_TYPES[$i]="dir"
    elif [[ -f "$mod_path" && "$mod_path" == *.zip ]]; then
        # Check zip content using unzip if available, else python
        if command -v unzip &> /dev/null; then
            if ! unzip -l "$mod_path" | grep -q "run.sh"; then
                 echo "Error: Mod zip file '$mod_path' must contain 'run.sh'."
                 exit 1
            fi
        else
             # Fallback to python for checking zip content
             if ! python3 -c "import zipfile, sys; sys.exit(0 if 'run.sh' in zipfile.ZipFile(sys.argv[1]).namelist() else 1)" "$mod_path"; then
                 echo "Error: Mod zip file '$mod_path' must contain 'run.sh'."
                 exit 1
             fi
        fi
        MOD_TYPES[$i]="zip"
    else
        echo "Error: --apply-mod '$mod_path' must be a directory or a .zip file."
        exit 1
    fi
    MOD_PATHS[$i]=$(realpath "$mod_path")
done

# --- Auto-Detection Logic ---
# Source autodiscover module
source "$(dirname "$0")/autodiscover.sh"

if [[ "$SOLO_MODE" == "true" ]]; then
    if [[ -n "$NODES_ARG" ]]; then
        echo "Error: --solo is incompatible with -n/--nodes."
        exit 1
    fi
    # Solo mode: skip node detection, just get local IP
    LOCAL_IP="127.0.0.1"
    NODES_ARG="$LOCAL_IP"
    PEER_NODES=()
    echo "Solo mode enabled. Skipping node detection."
else
    # Perform auto-detection
    detect_interfaces || exit 1
    detect_nodes || exit 1
fi

if [[ -z "$NODES_ARG" ]]; then
    echo "Error: Nodes argument (-n) is mandatory or could not be auto-detected."
    usage
fi

# Split nodes into array
IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"

if [[ "$SOLO_MODE" != "true" ]]; then
    # Detect Head IP (Local IP)
    detect_local_ip || exit 1
fi

HEAD_IP="$LOCAL_IP"

# Verify HEAD_IP is in ALL_NODES
FOUND_HEAD=false
for ip in "${ALL_NODES[@]}"; do
    ip=$(echo "$ip" | xargs)
    if [[ "$ip" == "$HEAD_IP" ]]; then
        FOUND_HEAD=true
        break
    fi
done

if [ "$FOUND_HEAD" = false ]; then
    echo "Error: Local IP ($HEAD_IP) is not in the list of nodes ($NODES_ARG)."
    exit 1
fi

# Implicit Solo Mode Detection
if [[ "$SOLO_MODE" == "false" && ${#PEER_NODES[@]} -eq 0 ]]; then
    echo "Only local node detected/configured. Activating solo mode (no Ray cluster)."
    SOLO_MODE="true"
fi

if [[ "$NO_RAY_MODE" == "true" && "$SOLO_MODE" == "true" ]]; then
    echo "Warning: Only one node detected; --no-ray has no effect in solo mode. Proceeding normally."
    NO_RAY_MODE="false"
fi

echo "Head Node: $HEAD_IP"
echo "Worker Nodes: ${PEER_NODES[*]}"
echo "Container Name: $CONTAINER_NAME"
echo "Image Name: $IMAGE_NAME"
echo "Action: $ACTION"

# Check SSH connectivity to worker nodes
if [[ "$ACTION" == "start" || "$ACTION" == "exec" || "$CHECK_CONFIG" == "true" ]]; then
    if [ ${#PEER_NODES[@]} -gt 0 ]; then
        echo "Checking SSH connectivity to worker nodes..."
        for worker in "${PEER_NODES[@]}"; do
            # Retry SSH up to 3 times with longer timeout — AO5's degraded NVMe
            # makes sshd's PAM/userlist lookups slow under load.
            if ! { ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$worker" true 2>/dev/null \
                || sleep 3 && ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$worker" true 2>/dev/null \
                || sleep 5 && ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$worker" true 2>/dev/null; }; then
                echo "Error: Passwordless SSH to $worker failed (after 3 retries)."
                echo "  Please ensure SSH keys are configured and the host is reachable."
                exit 1
            fi
            echo "  SSH to $worker: OK"
        done
    fi
fi

if [[ "$CHECK_CONFIG" == "true" ]]; then
    echo "Configuration Check Complete."
    echo "  Image Name: $IMAGE_NAME"
    echo "  ETH Interface: $ETH_IF"
    echo "  IB Interface: $IB_IF"
    echo "  Docker Args: $DOCKER_ARGS"
    if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
         echo "  Mounting Cache Dirs: ${CACHE_DIRS_TO_CREATE[*]}"
    else
         echo "  Mounting Cache Dirs: (Disabled)"
    fi
    exit 0
fi

# Cleanup Function
cleanup() {
    # Remove traps to prevent nested cleanup
    trap - EXIT INT TERM HUP

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        echo "Cluster was already running when script started. Skipping cleanup."
        return
    fi

    echo ""
    echo "Stopping cluster..."
    
    # Stop Head
    echo "Stopping head node ($HEAD_IP)..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    
    # Stop Workers
    for worker in "${PEER_NODES[@]}"; do
        echo "Stopping worker node ($worker)..."
        ssh "$worker" "docker stop $CONTAINER_NAME" >/dev/null 2>&1 || true
    done
    
    echo "Cluster stopped."
}

# Handle 'stop' action
if [[ "$ACTION" == "stop" ]]; then
    cleanup
    exit 0
fi

# Handle 'status' action
if [[ "$ACTION" == "status" ]]; then
    echo "Checking status..."
    
    # Check Head
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is RUNNING."
        if [[ "$NO_RAY_MODE" == "false" ]]; then
            echo "--- Ray Status ---"
            docker exec "$CONTAINER_NAME" ray status || echo "Failed to get ray status."
            echo "------------------"
        fi
    else
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is NOT running."
    fi
    
    # Check Workers
    for worker in "${PEER_NODES[@]}"; do
        if ssh "$worker" "docker ps | grep -q '$CONTAINER_NAME'"; then
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is RUNNING."
        else
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is NOT running."
        fi
    done
    exit 0
fi

# Trap signals
# Only trap if we are NOT in daemon mode (container should persist in daemon mode)
if [[ "$DAEMON_MODE" == "false" ]]; then
    trap cleanup EXIT INT TERM HUP
fi

# Check if cluster is already running
check_cluster_running() {
    local running=false
    
    # Check Head
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Warning: Container '$CONTAINER_NAME' is already running on head node ($HEAD_IP)."
        running=true
    fi
    
    # Check Workers
    for worker in "${PEER_NODES[@]}"; do
        if ssh "$worker" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
             echo "Warning: Container '$CONTAINER_NAME' is already running on worker node ($worker)."
             running=true
        fi
    done
    
    if [[ "$running" == "true" ]]; then
        echo "Cluster containers are already running. Skipping launch."
        CLUSTER_WAS_RUNNING="true"
        return 0
    fi
}

# Apply Mod Function
apply_mod_to_container() {
    local node_ip="$1"
    local container="$2"
    local is_local="$3" # true/false
    local mod_path="$4"
    local mod_type="$5"

    local mod_name=$(basename "$mod_path")
    if [[ "$mod_type" == "zip" ]]; then
        mod_name="${mod_name%.*}"
    fi

    echo "Applying mod '$mod_name' to $node_ip..."

    # 1. Copy mod to node (if remote)
    local target_mod_path=""
    local remote_cleanup_path=""

    if [[ "$is_local" == "true" ]]; then
        target_mod_path="$mod_path"
    else
        # SCP to remote
        local remote_tmp="/tmp/vllm_mod_pkg_$(date +%s)_$RANDOM"
        echo "  Copying mod package to $node_ip:$remote_tmp..."
        
        # Create directory first to ensure consistent path structure
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node_ip" "mkdir -p $remote_tmp"
        remote_cleanup_path="$remote_tmp"
        
        if [[ "$mod_type" == "zip" ]]; then
             if ! scp -o BatchMode=yes -o StrictHostKeyChecking=no "$mod_path" "$node_ip:$remote_tmp/"; then
                echo "Error: Failed to copy mod to $node_ip"
                exit 1
             fi
             target_mod_path="$remote_tmp/$(basename "$mod_path")"
        else
             # Directory
             # Copy contents using wildcard to avoid creating a subdirectory
             if ! scp -r -o BatchMode=yes -o StrictHostKeyChecking=no "$mod_path"/* "$node_ip:$remote_tmp/"; then
                echo "Error: Failed to copy mod to $node_ip"
                exit 1
             fi
             target_mod_path="$remote_tmp"
        fi
    fi

    # 2. Copy into container
    local container_dest="/workspace/mods/$mod_name"
    
    # Command prefix for remote vs local
    local cmd_prefix=""
    if [[ "$is_local" == "false" ]]; then
        cmd_prefix="ssh -o BatchMode=yes -o StrictHostKeyChecking=no $node_ip"
    fi

    # Create workspace in container
    $cmd_prefix docker exec "$container" mkdir -p "$container_dest"

    if [[ "$mod_type" == "zip" ]]; then
        local zip_name=$(basename "$mod_path")
        echo "  Copying zip to container..."
        $cmd_prefix docker cp "$target_mod_path" "$container:$container_dest/$zip_name"
        
        # Unzip in container using python
        echo "  Extracting zip..."
        local py_unzip="import zipfile, sys; zipfile.ZipFile(sys.argv[1], 'r').extractall(sys.argv[2])"
        if [[ "$is_local" == "true" ]]; then
            docker exec "$container" python3 -c "$py_unzip" "$container_dest/$zip_name" "$container_dest"
        else
            $cmd_prefix docker exec "$container" python3 -c "\"$py_unzip\"" "$container_dest/$zip_name" "$container_dest"
        fi
    else
        # Directory
        echo "  Copying directory content to container..."
        if [[ "$is_local" == "true" ]]; then
             docker cp "$mod_path/." "$container:$container_dest/"
        else
             # For remote, we copied contents to $target_mod_path.
             # We want to copy contents of $target_mod_path to $container_dest.
             $cmd_prefix docker cp "$target_mod_path/." "$container:$container_dest/"
        fi
    fi

    # 3. Run run.sh
    echo "  Running patch script on $node_ip..."

    local local_exec_cmd="export WORKSPACE_DIR=\$PWD && cd $container_dest && chmod +x run.sh && ./run.sh"
    local remote_exec_cmd="export WORKSPACE_DIR=\\\$PWD && cd $container_dest && chmod +x run.sh && ./run.sh"
    local ret_code=0

    if [[ "$is_local" == "true" ]]; then
        docker exec "$container" bash -c "$local_exec_cmd"
        ret_code=$?
    else
        $cmd_prefix docker exec "$container" bash -c "\"$remote_exec_cmd\""
        ret_code=$?
    fi

    if [[ $ret_code -ne 0 ]]; then
        echo "Error: Patch script failed on $node_ip"
        # We should probably stop the cluster here or at least fail hard
        exit 1
    fi

    # 4. Cleanup remote temp
    if [[ "$is_local" == "false" ]]; then
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node_ip" "rm -rf $remote_cleanup_path"
    fi
}

# Build a patched copy of the launch script on the host for a specific node.
# Strips --distributed-executor-backend and appends multi-node args.
# Prints the path of the temp file (caller must delete it).
make_node_script() {
    local script_path="$1"; local nnodes="$2"; local node_rank="$3"; local master_addr="$4"
    local extra="--nnodes $nnodes --node-rank $node_rank --master-addr $master_addr --master-port $MASTER_PORT"
    [[ "$node_rank" -gt 0 ]] && extra="$extra --headless"

    local tmp; tmp=$(mktemp /tmp/vllm_node_script_XXXXXX.sh)
    # Remove just the flag and its value (not the whole line), then filter empty/backslash-only lines
    sed 's/--distributed-executor-backend[[:space:]]*[^[:space:]]*//' "$script_path" | \
        grep -Ev '^[[:space:]\\]*$' > "$tmp"
    # Strip trailing backslash from last line before appending multi-node args
    sed -i "$ s/[[:space:]]*\\\\[[:space:]]*$//" "$tmp"
    sed -i "$ s/$/ $extra/" "$tmp"
    chmod +x "$tmp"
    echo "$tmp"
}

# Copy a script file into a local container as /workspace/exec-script.sh
copy_script_to_container() {
    local container="$1"; local script_path="$2"; local label="${3:-node}"
    echo "Copying launch script to $label..."
    docker cp "$script_path" "$container:/workspace/exec-script.sh" || { echo "Error: docker cp to $label failed"; exit 1; }
    docker exec "$container" chmod +x /workspace/exec-script.sh
}

# Copy a script file to a remote container via scp + docker cp
copy_script_to_worker() {
    local worker_ip="$1"; local container="$2"; local script_path="$3"
    echo "Copying launch script to worker $worker_ip..."
    local remote_tmp="/tmp/vllm_script_$(date +%s)_$RANDOM.sh"
    scp -o BatchMode=yes -o StrictHostKeyChecking=no "$script_path" "$worker_ip:$remote_tmp" || { echo "Error: scp to $worker_ip failed"; exit 1; }
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker_ip" \
        "docker cp $remote_tmp $container:/workspace/exec-script.sh && \
         docker exec $container chmod +x /workspace/exec-script.sh && \
         rm -f $remote_tmp" || { echo "Error: docker cp to worker $worker_ip failed"; exit 1; }
}

# Build -e KEY=VALUE flags for a given node IP (used in docker run and docker exec)
get_env_flags() {
    local node_ip="$1"
    printf -- '-e %s ' \
        "VLLM_HOST_IP=$node_ip" \
        "RAY_NODE_IP_ADDRESS=$node_ip" \
        "RAY_OVERRIDE_NODE_IP_ADDRESS=$node_ip" \
        "MN_IF_NAME=$ETH_IF" \
        "UCX_NET_DEVICES=$ETH_IF" \
        "NCCL_SOCKET_IFNAME=$ETH_IF" \
        "NCCL_IB_HCA=$IB_IF" \
        "NCCL_IB_DISABLE=0" \
        "OMPI_MCA_btl_tcp_if_include=$ETH_IF" \
        "GLOO_SOCKET_IFNAME=$ETH_IF" \
        "TP_SOCKET_IFNAME=$ETH_IF" \
        "RAY_memory_monitor_refresh_ms=0" \
        "RAY_num_prestart_python_workers=0" \
        "RAY_object_store_memory=1073741824"
}

# Start Ray head node inside the container
start_ray_head() {
    local container="$1"
    echo "Starting Ray HEAD node on $HEAD_IP..."
    docker exec -d "$container" bash -c \
        "ray start --block --head --port $MASTER_PORT --object-store-memory 1073741824 --num-cpus 2 \
         --node-ip-address $HEAD_IP --include-dashboard=false --disable-usage-stats \
         >> /proc/1/fd/1 2>&1"
}

# Start Ray worker node inside the container on a remote host
start_ray_worker() {
    local worker_ip="$1"; local container="$2"
    echo "Starting Ray WORKER node on $worker_ip..."
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker_ip" \
        "docker exec -d $container bash -c \
         'ray start --block --object-store-memory 1073741824 --num-cpus 2 --disable-usage-stats \
          --address=$HEAD_IP:$MASTER_PORT --node-ip-address $worker_ip >> /proc/1/fd/1 2>&1'"
}

# Start Cluster Function
start_cluster() {
    check_cluster_running

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        return
    fi

    # Build docker run arguments based on mode
    local docker_args_common="--gpus all -d --rm --network host --name $CONTAINER_NAME $DOCKER_ARGS $IMAGE_NAME"
    local docker_caps_args=""
    local docker_resource_args=""

    if [[ "$NON_PRIVILEGED_MODE" == "true" ]]; then
        echo "Running in non-privileged mode..."
        docker_caps_args="--cap-add=IPC_LOCK"
        docker_resource_args="--shm-size=${SHM_SIZE_GB}g --device=/dev/infiniband --memory ${MEM_LIMIT_GB}g --memory-swap ${MEM_SWAP_LIMIT_GB}g --pids-limit ${PIDS_LIMIT}"
    else
        docker_caps_args="--privileged"
        docker_resource_args="--ipc=host"
    fi

    # Start Head Node
    echo "Starting Head Node on $HEAD_IP..."
    if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
        for dir in "${CACHE_DIRS_TO_CREATE[@]}"; do
            mkdir -p "$dir"
        done
    fi
    docker run $docker_caps_args $docker_resource_args \
        $(get_env_flags "$HEAD_IP") $docker_args_common sleep infinity

    # Start Worker Nodes
    for worker in "${PEER_NODES[@]}"; do
        echo "Starting Worker Node on $worker..."
        if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
            ssh "$worker" "mkdir -p ${CACHE_DIRS_TO_CREATE[*]}"
        fi
        local docker_run_cmd="docker run $docker_caps_args $docker_resource_args $(get_env_flags "$worker") $docker_args_common"
        ssh "$worker" "$docker_run_cmd sleep infinity"
    done

    # Apply mods (containers are idle — no mod_done sync needed)
    if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
        echo "Applying modifications to cluster nodes..."
        for i in "${!MOD_PATHS[@]}"; do
            apply_mod_to_container "$HEAD_IP" "$CONTAINER_NAME" "true" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
        done
        for worker in "${PEER_NODES[@]}"; do
            for i in "${!MOD_PATHS[@]}"; do
                apply_mod_to_container "$worker" "$CONTAINER_NAME" "false" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
            done
        done
    fi

    # Copy (and patch for no-ray) launch script
    if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
        local total_nodes=$(( 1 + ${#PEER_NODES[@]} ))
        if [[ "$NO_RAY_MODE" == "true" ]]; then
            # Build per-node patched scripts on the host, then copy
            local head_script; head_script=$(make_node_script "$LAUNCH_SCRIPT_PATH" "$total_nodes" "0" "$HEAD_IP")
            copy_script_to_container "$CONTAINER_NAME" "$head_script" "head node ($HEAD_IP)"
            rm -f "$head_script"

            local rank=1
            for worker in "${PEER_NODES[@]}"; do
                local worker_script; worker_script=$(make_node_script "$LAUNCH_SCRIPT_PATH" "$total_nodes" "$rank" "$HEAD_IP")
                copy_script_to_worker "$worker" "$CONTAINER_NAME" "$worker_script"
                rm -f "$worker_script"
                (( rank++ ))
            done
        else
            copy_script_to_container "$CONTAINER_NAME" "$LAUNCH_SCRIPT_PATH" "head node"
        fi
    fi

    # Patch Ray's hardcoded 30s raylet_start_wait_time_s to 600s.
    # On the DGX Spark cluster, AO5's NVMe is degraded and dashboard_agent
    # bootstrap can exceed 30s, causing the python ray-start wrapper to die
    # with "The current node timed out during startup", orphaning the raylet
    # as <defunct> and silently dropping the node from the placement group.
    # See node.py:411 — env var doesn't help, value is hardcoded.
    patch_ray_timeout_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local cmd="docker exec $container python3 -c \"import re,sys; p='/usr/local/lib/python3.12/dist-packages/ray/_private/node.py'; s=open(p).read(); n=s.replace('raylet_start_wait_time_s = 30','raylet_start_wait_time_s = 600'); open(p,'w').write(n); sys.stdout.write('PATCHED' if s!=n else 'NOOP')\""
        if [[ "$is_local" == "true" ]]; then
            eval $cmd
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "$cmd"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching Ray raylet_start_wait_time_s 30→600 in all containers..."
        patch_ray_timeout_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_ray_timeout_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi

    # Patch vLLM's compressed_tensors MoE dispatch to bypass Marlin on GB10 sm_121.
    # Default vLLM picks Marlin for WNA16 MoE when check_moe_marlin_supports_layer()
    # returns True, but Marlin's moe_wna16_marlin_gemm kernel crashes with illegal
    # memory access on sm_121 (GB10). The fallback CompressedTensorsWNA16MoEMethod
    # (Triton-based) works correctly. Force the fallback by replacing the
    # check_moe_marlin_supports_layer(...) call with True (so 'not True' = False
    # in the if-condition, but we want True path... read the source: 'if NOT check'
    # selects fallback; 'else' selects Marlin. So replacing the function call with
    # False makes 'not False' = True = always pick fallback. That's the fix.
    patch_moe_dispatch_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local cmd="docker exec $container python3 -c \"import os, sys; cand=['/opt/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py']; p=next((c for c in cand if os.path.exists(c)), None); sys.stdout.write('SKIP-not-found' if p is None else ''); s=open(p).read() if p else ''; n=s.replace('not check_moe_marlin_supports_layer(layer, group_size)', 'True  # patched: force non-Marlin path on sm_121') if p else ''; open(p,'w').write(n) if p and s!=n else None; sys.stdout.write('PATCHED' if (p and s!=n) else ('NOOP' if p else ''))\""
        if [[ "$is_local" == "true" ]]; then
            eval $cmd
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "$cmd"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching vLLM compressed_tensors MoE dispatch (force Triton, bypass Marlin sm_121 bug)..."
        patch_moe_dispatch_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_moe_dispatch_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi

    # Patch vLLM api_server.py to skip the startup reset_mm_cache() call.
    # On multi-node K2.6 with this hardware, the first NCCL collective in
    # serving phase (which is what reset_mm_cache triggers) reliably hangs/dies
    # with ActorDiedError. The reset is only clearing dummy data left from
    # memory profiling — skipping it leaves a small chunk of dummy multimodal
    # data resident until first real request preempts it. Acceptable trade-off
    # vs. the engine refusing to start.
    patch_skip_reset_mm_cache_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local cmd="docker exec $container python3 -c \"import os, sys; cand=['/opt/vllm/vllm/entrypoints/openai/api_server.py','/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py']; p=next((c for c in cand if os.path.exists(c)), None); s=open(p).read() if p else ''; n=s.replace('await async_llm.reset_mm_cache()', 'pass  # patched: skip on multi-node K2.6 first NCCL collective hangs') if p else ''; open(p,'w').write(n) if p and s!=n else None; sys.stdout.write('PATCHED' if (p and s!=n) else ('NOOP' if p else 'SKIP-not-found'))\""
        if [[ "$is_local" == "true" ]]; then
            eval $cmd
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "$cmd"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching vLLM api_server.py to skip startup reset_mm_cache (avoid first-NCCL-collective hang on K2.6)..."
        patch_skip_reset_mm_cache_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_skip_reset_mm_cache_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi

    # Patch nvidia_cutlass_dsl BlockScaledMmaOp.admissible_archs to include sm_120a + sm_121a.
    # The default list only has sm_100a and sm_103a — refusing to dispatch on GB10 (sm_121).
    # Per NVIDIA Forum DGX Spark recipe (Qwen3.5-NVFP4 success post): adding GB10's archs
    # is the path that works for cutlass NVFP4 MoE on GB10. Required for any modelopt_fp4
    # quantized model with --moe-backend cutlass.
    patch_cute_arch_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local cmd="docker exec $container python3 -c \"import os, sys; cand=['/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py','/opt/vllm/.deps/cutlass-src/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py']; p=next((c for c in cand if os.path.exists(c)), None); s=open(p).read() if p else ''; old='admissible_archs = [\n        Arch.sm_100a,\n        Arch.sm_103a,\n    ]'; new='admissible_archs = [\n        Arch.sm_100a,\n        Arch.sm_103a,\n        Arch.sm_120a,  # patched: GB10 / DGX Spark\n        Arch.sm_121a,  # patched: GB10 / DGX Spark\n    ]'; n=s.replace(old, new) if p else ''; open(p,'w').write(n) if p and s!=n else None; sys.stdout.write('PATCHED' if (p and s!=n) else ('NOOP' if p else 'SKIP-not-found'))\""
        if [[ "$is_local" == "true" ]]; then
            eval $cmd
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "$cmd"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching cute/mma.py BlockScaledMmaOp.admissible_archs to include sm_120a + sm_121a (GB10 NVFP4)..."
        patch_cute_arch_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_cute_arch_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi

    # Patch vLLM b12x FlashInferB12xExperts.apply() + flashinfer kernels (state_E rewrite).
    patch_b12x_ep_slice_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local script_path="$HOME/spark-vllm-docker-main/patch_b12x_ep_slice.py"
        if [[ "$is_local" == "true" ]]; then
            docker cp "$script_path" "$container":/tmp/patch_b12x_ep_slice.py
            docker exec "$container" python3 /tmp/patch_b12x_ep_slice.py
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "docker exec -i $container tee /tmp/patch_b12x_ep_slice.py >/dev/null && docker exec $container python3 /tmp/patch_b12x_ep_slice.py" < "$script_path"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching vLLM b12x + flashinfer kernels (state_E v5 — vllm#40082)..."
        patch_b12x_ep_slice_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_b12x_ep_slice_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi


    # MIMO-marlin-chunked-repack-patch
    # Patch vLLM Marlin NVFP4 MoE weight repack to stream into a preallocated
    # output tensor instead of accumulating 48 expert tensors + torch.cat.
    # Drops peak memory by ~36 GiB so head node (extra Ray/API/EngineCore
    # overhead) fits Marlin process_weights on GB10 121 GiB unified mem.
    # See ~/Documents/Claude/Projects/AO/DEPLOYMENT_RSI_LOG.md Cycle 12.
    patch_marlin_chunked_repack_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local script_path="$HOME/spark-vllm-docker-main/patch_marlin_chunked_repack.py"
        if [[ "$is_local" == "true" ]]; then
            docker cp "$script_path" "$container":/tmp/patch_marlin_chunked_repack.py
            docker exec "$container" python3 /tmp/patch_marlin_chunked_repack.py
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "docker exec -i $container tee /tmp/patch_marlin_chunked_repack.py >/dev/null && docker exec $container python3 /tmp/patch_marlin_chunked_repack.py" < "$script_path"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching vLLM Marlin NVFP4 MoE chunked repack (head-OOM fix)..."
        patch_marlin_chunked_repack_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_marlin_chunked_repack_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi


    # MIMO-modelopt-mxfp8-dispatch-patch
    # Wire MXFP8 into ModelOptMixedPrecisionConfig.get_quant_method dispatch.
    # festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8 tags attention layers as
    # quant_algo: "MXFP8" but vLLM's modelopt_mixed code only branches on
    # "FP8" and "NVFP4" -- MXFP8 falls through to UnquantizedLinearMethod
    # and attention runs on garbage values. See DEPLOYMENT_RSI_LOG.md C15.
    patch_modelopt_mxfp8_dispatch_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local script_path="$HOME/spark-vllm-docker-main/patch_modelopt_mxfp8_dispatch.py"
        if [[ "$is_local" == "true" ]]; then
            docker cp "$script_path" "$container":/tmp/patch_modelopt_mxfp8_dispatch.py
            docker exec "$container" python3 /tmp/patch_modelopt_mxfp8_dispatch.py
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "docker exec -i $container tee /tmp/patch_modelopt_mxfp8_dispatch.py >/dev/null && docker exec $container python3 /tmp/patch_modelopt_mxfp8_dispatch.py" < "$script_path"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching vLLM modelopt_mixed MXFP8 dispatch (festr2 attention fix)..."
        patch_modelopt_mxfp8_dispatch_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_modelopt_mxfp8_dispatch_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi


    # MIMO-modelopt-mxfp8-scale-name-patch
    # Fix the actual MXFP8 weight-load bug: ModelOptMxFp8LinearMethod registers
    # the scale parameter as `weight_scale`, but festr2 stores it as
    # `weight_scale_inv` in safetensors. Without this rename, the scale param
    # never gets populated (loader matches by name) and dequant uses zeros ->
    # garbage logits regardless of which dequant kernel runs. See C15/C16
    # diff in DEPLOYMENT_RSI_LOG.md.
    patch_modelopt_mxfp8_scale_name_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local script_path="$HOME/spark-vllm-docker-main/patch_modelopt_mxfp8_scale_name.py"
        if [[ "$is_local" == "true" ]]; then
            docker cp "$script_path" "$container":/tmp/patch_modelopt_mxfp8_scale_name.py
            docker exec "$container" python3 /tmp/patch_modelopt_mxfp8_scale_name.py
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "docker exec -i $container tee /tmp/patch_modelopt_mxfp8_scale_name.py >/dev/null && docker exec $container python3 /tmp/patch_modelopt_mxfp8_scale_name.py" < "$script_path"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching ModelOptMxFp8 weight_scale -> weight_scale_inv rename (festr2 scale-name fix)..."
        patch_modelopt_mxfp8_scale_name_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_modelopt_mxfp8_scale_name_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi


    # MIMO-mimo-qkv-split-patch
    # Fix MiMo-V2 fused qkv_proj loader: original chunk(tp,dim=0)[rank] is wrong
    # for festr2 layout where qkv is fused [Q|K|V] in dim 0 (Q=24576, K=1536,
    # V=1024 for full attention). Split by Q/K/V sizes and use proper loader
    # with shard_id for each section.
    patch_mimo_qkv_split_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local script_path="$HOME/spark-vllm-docker-main/patch_mimo_qkv_split.py"
        if [[ "$is_local" == "true" ]]; then
            docker cp "$script_path" "$container":/tmp/patch_mimo_qkv_split.py
            docker exec "$container" python3 /tmp/patch_mimo_qkv_split.py
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "docker exec -i $container tee /tmp/patch_mimo_qkv_split.py >/dev/null && docker exec $container python3 /tmp/patch_mimo_qkv_split.py" < "$script_path"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching MiMo-V2 fused qkv_proj loader Q/K/V split (festr2 fused-qkv layout fix)..."
        patch_mimo_qkv_split_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_mimo_qkv_split_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi


    # MIMO-mimo-v2-mtp-qkv-split-patch
    # Same fused-qkv-split bug exists in vLLM's mimo_v2_mtp.py (MTP draft
    # head loader). Without this, MTP attention runs with Q values in K/V
    # slots on tp_size-1 of tp_size ranks, draft tokens are near-random,
    # and acceptance hovers at 3-24% instead of the 70%+ a healthy MTP
    # head should achieve.
    patch_mimo_v2_mtp_qkv_split_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local script_path="$HOME/spark-vllm-docker-main/patch_mimo_v2_mtp_qkv_split.py"
        if [[ "$is_local" == "true" ]]; then
            docker cp "$script_path" "$container":/tmp/patch_mimo_v2_mtp_qkv_split.py
            docker exec "$container" python3 /tmp/patch_mimo_v2_mtp_qkv_split.py
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "docker exec -i $container tee /tmp/patch_mimo_v2_mtp_qkv_split.py >/dev/null && docker exec $container python3 /tmp/patch_mimo_v2_mtp_qkv_split.py" < "$script_path"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching MiMo-V2-MTP fused qkv_proj loader Q/K/V split (MTP head fix)..."
        patch_mimo_v2_mtp_qkv_split_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_mimo_v2_mtp_qkv_split_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi


    # Patch CUTLASS_MLA + FLASHMLA + FLASH_ATTN_MLA to accept sm_12x (GB10).
    # Default supports_compute_capability checks major == 10 (CUTLASS_MLA) or
    # major == 9 / [9,10] (FLASH_ATTN_MLA / FLASHMLA). GB10 is sm_121 (major 12).
    # The TRITON_MLA fallback hits a shared-memory shortage on GB10
    # (Required: 102400, Hardware limit: 101376) — same 1KB shortage as the
    # K2.6-original FlashInfer FP16 MoE issue. Patching the optimized backends
    # to accept major==12 lets vLLM pick CUTLASS_MLA (Blackwell-class kernel)
    # instead of falling back to broken Triton.
    patch_mla_compute_capability_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        # Three separate single-line patches (multi-line heredoc gets corrupted through ssh+docker exec).
        # Each tries /opt/vllm/vllm first (main-sm12x image) then /usr/local/.../vllm (eugr image).
        local cmd1="docker exec $container python3 -c \"import os; cand=['/opt/vllm/vllm/v1/attention/backends/mla/cutlass_mla.py','/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/cutlass_mla.py']; p=next((c for c in cand if os.path.exists(c)), None); s=open(p).read() if p else ''; n=s.replace('return capability.major == 10', 'return capability.major == 10 or capability.major == 12') if p else ''; open(p,'w').write(n) if p and s!=n else None; print('cutlass_mla:'+('PATCHED' if (p and s!=n) else ('NOOP' if p else 'SKIP-not-found')))\""
        local cmd2="docker exec $container python3 -c \"import os; cand=['/opt/vllm/vllm/v1/attention/backends/mla/flashmla.py','/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla.py']; p=next((c for c in cand if os.path.exists(c)), None); s=open(p).read() if p else ''; n=s.replace('return capability.major in [9, 10]', 'return capability.major in [9, 10, 12]') if p else ''; open(p,'w').write(n) if p and s!=n else None; print('flashmla:'+('PATCHED' if (p and s!=n) else ('NOOP' if p else 'SKIP-not-found')))\""
        local cmd3="docker exec $container python3 -c \"import os; cand=['/opt/vllm/vllm/v1/attention/backends/mla/flashattn_mla.py','/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashattn_mla.py']; p=next((c for c in cand if os.path.exists(c)), None); s=open(p).read() if p else ''; n=s.replace('return capability.major == 9', 'return capability.major == 9 or capability.major == 12') if p else ''; open(p,'w').write(n) if p and s!=n else None; print('flashattn_mla:'+('PATCHED' if (p and s!=n) else ('NOOP' if p else 'SKIP-not-found')))\""
        # NOTE: previously also patched cuda.py _get_backend_priorities to add CUTLASS_MLA
        # for major == 12, but the sm100_cutlass_mla_decode C++ kernel rejects sm_121 at
        # runtime ("Error Internal"). Reverted — TRITON_MLA path with BLOCK=16 patch is
        # the working route on GB10.
        if [[ "$is_local" == "true" ]]; then
            eval $cmd1; eval $cmd2; eval $cmd3
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "$cmd1; $cmd2; $cmd3"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching MLA backends (cutlass_mla / flashmla / flashattn_mla) to accept sm_12x (GB10)..."
        patch_mla_compute_capability_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_mla_compute_capability_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi

    # Patch TRITON_MLA decode kernel block size for GB10 sm_121 shmem limit.
    # File: vllm/v1/attention/ops/triton_decode_attention.py
    # vLLM already has a workaround for AMD MI3xx (same shmem constraint as GB10):
    #     if is_hip_ and Lk >= 576: BLOCK = 16
    # GB10 has 101.4 KB shared memory per SM; the default BLOCK=32 with Lk=576
    # (K2.6 MLA head_size) requires 102.4 KB → triton.runtime.errors.OutOfResources.
    # Removing the is_hip_ gate so the workaround also applies to GB10.
    patch_triton_mla_block_size_in_container() {
        local target="$1"; local container="$2"; local is_local="$3"
        local cmd="docker exec $container python3 -c \"import os; cand=['/opt/vllm/vllm/v1/attention/ops/triton_decode_attention.py','/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_decode_attention.py']; p=next((c for c in cand if os.path.exists(c)), None); s=open(p).read() if p else ''; n=s.replace('if is_hip_ and Lk >= 576:', 'if Lk >= 576:  # patched: GB10 sm_121 has same shmem limit as MI3xx') if p else ''; open(p,'w').write(n) if p and s!=n else None; print('triton_decode_attention:'+('PATCHED' if (p and s!=n) else ('NOOP' if p else 'SKIP-not-found')))\""
        if [[ "$is_local" == "true" ]]; then
            eval $cmd
        else
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$target" "$cmd"
        fi
    }
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        echo "Patching TRITON_MLA decode BLOCK 32→16 for Lk>=576 (GB10 shmem fix)..."
        patch_triton_mla_block_size_in_container "$HEAD_IP" "$CONTAINER_NAME" "true" >/dev/null
        for worker in "${PEER_NODES[@]}"; do
            patch_triton_mla_block_size_in_container "$worker" "$CONTAINER_NAME" "false" >/dev/null
        done
    fi

    # Start Ray cluster (unless solo or no-ray)
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        start_ray_head "$CONTAINER_NAME"
        for worker in "${PEER_NODES[@]}"; do
            start_ray_worker "$worker" "$CONTAINER_NAME"
        done
        wait_for_cluster
    else
        sleep 2
    fi
}

# Wait for Cluster Readiness
wait_for_cluster() {
    echo "Waiting for cluster to be ready..."
    local retries=30
    local count=0
    
    while [[ $count -lt $retries ]]; do
        # Check if ray is responsive
        if docker exec "$CONTAINER_NAME" ray status >/dev/null 2>&1; then
             echo "Cluster head is responsive."
             # Give workers a moment to connect
             sleep 5
             return 0
        fi
        
        sleep 2
        ((count++))
    done
    
    echo "Timeout waiting for cluster to start."
    exit 1
}

# Execute command on head node (daemon or interactive)
_exec_on_head() {
    local cmd="$1"
    if [[ "$DAEMON_MODE" == "true" ]]; then
        docker exec -d "$CONTAINER_NAME" bash -c "$cmd >> /proc/1/fd/1 2>&1"
        echo "Command dispatched in background (Daemon mode). Container: $CONTAINER_NAME"
    else
        if [ -t 0 ]; then DOCKER_EXEC_FLAGS="-it"; else DOCKER_EXEC_FLAGS="-i"; fi
        docker exec $DOCKER_EXEC_FLAGS "$CONTAINER_NAME" bash -c "$cmd"
    fi
}

# Execute a no-ray multi-node command: workers (background) then head
exec_no_ray_cluster() {
    local base_cmd="$1"
    local total_nodes=$(( 1 + ${#PEER_NODES[@]} ))

    # Launch workers first (always background)
    local rank=1
    for worker in "${PEER_NODES[@]}"; do
        local worker_cmd
        if [[ "$LAUNCH_SCRIPT_MODE" == "true" ]]; then
            worker_cmd="$base_cmd"  # script already patched per-node in start_cluster()
        else
            local clean
            clean=$(echo "$base_cmd" | sed 's/--distributed-executor-backend[[:space:]]*[^[:space:]]*//')
            worker_cmd="$clean --nnodes $total_nodes --node-rank $rank --master-addr $HEAD_IP --master-port $MASTER_PORT --headless"
        fi
        echo "Launching worker (rank $rank) on $worker..."
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker" \
            "docker exec -d $CONTAINER_NAME bash -c \"$worker_cmd >> /proc/1/fd/1 2>&1\""
        (( rank++ ))
    done

    # Launch head (rank 0) last
    local head_cmd
    if [[ "$LAUNCH_SCRIPT_MODE" == "true" ]]; then
        head_cmd="$base_cmd"
    else
        local clean
        clean=$(echo "$base_cmd" | sed 's/--distributed-executor-backend[[:space:]]*[^[:space:]]*//')
        head_cmd="$clean --nnodes $total_nodes --node-rank 0 --master-addr $HEAD_IP --master-port $MASTER_PORT"
    fi

    echo "Executing command on head node (rank 0): $head_cmd"
    if [[ "$DAEMON_MODE" == "true" ]]; then
        docker exec -d "$CONTAINER_NAME" bash -c "$head_cmd >> /proc/1/fd/1 2>&1"
        echo "Command dispatched in background (Daemon mode). Container: $CONTAINER_NAME"
    else
        if [ -t 0 ]; then DOCKER_EXEC_FLAGS="-it"; else DOCKER_EXEC_FLAGS="-i"; fi
        docker exec $DOCKER_EXEC_FLAGS "$CONTAINER_NAME" bash -c "$head_cmd"
    fi
}

if [[ "$ACTION" == "exec" ]]; then
    start_cluster
    echo "Executing command: $COMMAND_TO_RUN"

    if [[ "$NO_RAY_MODE" == "true" && ${#PEER_NODES[@]} -gt 0 ]]; then
        if [[ "$LAUNCH_SCRIPT_MODE" == "true" ]] || echo "$COMMAND_TO_RUN" | grep -q "vllm serve"; then
            exec_no_ray_cluster "$COMMAND_TO_RUN"
        else
            _exec_on_head "$COMMAND_TO_RUN"
        fi
    else
        _exec_on_head "$COMMAND_TO_RUN"
    fi
elif [[ "$ACTION" == "start" ]]; then
    start_cluster
    if [[ "$DAEMON_MODE" == "true" ]]; then
        echo "Cluster started in background (Daemon mode)."
    else
        echo "Cluster started. Tailing logs from head node..."
        echo "Press Ctrl+C to stop the cluster."
        docker logs -f "$CONTAINER_NAME" &
        wait $!
    fi
fi
